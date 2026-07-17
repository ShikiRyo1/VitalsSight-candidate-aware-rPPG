from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Mapping

import jwt
from jwt import PyJWKClient

from src.product.identity import (
    DEFAULT_ORGANIZATION_ID,
    DEFAULT_USER_ID,
    IdentityContext,
    identity_from_claims,
    local_identity,
    normalize_identifier,
    organizations_from_claims,
)


class AuthError(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AuthSettings:
    mode: str = "disabled"
    issuer: str = ""
    audience: str = ""
    client_id: str = ""
    jwks_url: str = ""
    shared_secret: str = ""
    algorithms: tuple[str, ...] = ("RS256",)
    leeway_seconds: int = 30
    allow_dev_identity_headers: bool = False

    @classmethod
    def from_env(cls) -> "AuthSettings":
        mode = os.getenv("VITALSSIGHT_AUTH_MODE", "disabled").strip().lower()
        if mode not in {"disabled", "required"}:
            raise ValueError("VITALSSIGHT_AUTH_MODE must be disabled or required")
        shared_secret = os.getenv("VITALSSIGHT_AUTH_SHARED_SECRET", "").strip()
        algorithms = tuple(
            item.strip()
            for item in os.getenv(
                "VITALSSIGHT_AUTH_ALGORITHMS",
                "HS256" if shared_secret else "RS256",
            ).split(",")
            if item.strip()
        )
        settings = cls(
            mode=mode,
            issuer=os.getenv("VITALSSIGHT_AUTH_ISSUER", "").rstrip("/"),
            audience=os.getenv("VITALSSIGHT_AUTH_AUDIENCE", "").strip(),
            client_id=os.getenv("VITALSSIGHT_AUTH_CLIENT_ID", "").strip(),
            jwks_url=os.getenv("VITALSSIGHT_AUTH_JWKS_URL", "").strip(),
            shared_secret=shared_secret,
            algorithms=algorithms,
            leeway_seconds=int(os.getenv("VITALSSIGHT_AUTH_LEEWAY_SECONDS", "30")),
            allow_dev_identity_headers=_truthy(
                os.getenv("VITALSSIGHT_ALLOW_DEV_IDENTITY_HEADERS")
            ),
        )
        if settings.mode == "required":
            if not settings.issuer or not settings.audience:
                raise ValueError(
                    "Required auth mode needs VITALSSIGHT_AUTH_ISSUER and VITALSSIGHT_AUTH_AUDIENCE"
                )
            if not settings.shared_secret and not settings.jwks_url:
                raise ValueError(
                    "Required auth mode needs VITALSSIGHT_AUTH_JWKS_URL or a test-only shared secret"
                )
        return settings

    def public_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "issuer": self.issuer,
            "audience": self.audience,
            "client_id": self.client_id,
            "algorithms": list(self.algorithms),
            "dev_identity_headers": self.allow_dev_identity_headers,
        }


class IdentityResolver:
    """Resolve disabled-local or verified OIDC identities without storing passwords."""

    def __init__(self, settings: AuthSettings | None = None) -> None:
        self.settings = settings or AuthSettings.from_env()
        self._jwks_client = (
            PyJWKClient(self.settings.jwks_url, cache_keys=True)
            if self.settings.jwks_url and not self.settings.shared_secret
            else None
        )

    def resolve(
        self,
        authorization: str | None,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> IdentityContext:
        normalized_headers = {str(key).lower(): str(value) for key, value in (headers or {}).items()}
        if self.settings.mode == "disabled":
            if self.settings.allow_dev_identity_headers:
                roles = tuple(
                    item.strip()
                    for item in normalized_headers.get("x-vitalssight-roles", "").split(",")
                    if item.strip()
                )
                return local_identity(
                    organization_id=normalized_headers.get(
                        "x-vitalssight-organization", DEFAULT_ORGANIZATION_ID
                    ),
                    user_id=normalized_headers.get("x-vitalssight-user", DEFAULT_USER_ID),
                    display_name=normalized_headers.get(
                        "x-vitalssight-display-name", "Research operator"
                    ),
                    roles=roles or local_identity().roles,
                )
            return local_identity()

        token = self._bearer_token(authorization)
        claims = self._decode(token)
        return self.resolve_claims(
            claims,
            requested_organization=normalized_headers.get("x-vitalssight-organization", ""),
        )

    @staticmethod
    def _bearer_token(authorization: str | None) -> str:
        scheme, _, token = str(authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise AuthError(401, "A valid bearer token is required")
        return token.strip()

    def _decode(self, token: str) -> dict[str, Any]:
        try:
            if self.settings.shared_secret:
                key: Any = self.settings.shared_secret
            else:
                if self._jwks_client is None:
                    raise AuthError(503, "OIDC key discovery is unavailable")
                key = self._jwks_client.get_signing_key_from_jwt(token).key
            claims = jwt.decode(
                token,
                key,
                algorithms=list(self.settings.algorithms),
                audience=self.settings.audience,
                issuer=self.settings.issuer,
                leeway=self.settings.leeway_seconds,
                options={"require": ["exp", "sub", "iss", "aud"]},
            )
            return dict(claims)
        except AuthError:
            raise
        except jwt.ExpiredSignatureError as error:
            raise AuthError(401, "The access token has expired") from error
        except jwt.InvalidAudienceError as error:
            raise AuthError(401, "The access token audience is not accepted") from error
        except jwt.InvalidIssuerError as error:
            raise AuthError(401, "The access token issuer is not accepted") from error
        except jwt.PyJWTError as error:
            raise AuthError(401, "The access token could not be verified") from error

    def resolve_claims(
        self,
        claims: Mapping[str, Any],
        *,
        requested_organization: str = "",
    ) -> IdentityContext:
        organizations = organizations_from_claims(claims)
        requested = normalize_identifier(requested_organization, fallback="")
        if requested:
            if requested not in organizations:
                raise AuthError(403, "The requested organization is not present in the token")
            organization_id = requested
        elif len(organizations) == 1:
            organization_id = next(iter(organizations))
        elif not organizations:
            raise AuthError(403, "The token does not include an organization context")
        else:
            raise AuthError(403, "Select one organization for this request")
        try:
            return identity_from_claims(
                claims,
                organization_id=organization_id,
                auth_mode="oidc",
                client_id=self.settings.client_id,
            )
        except ValueError as error:
            raise AuthError(401, str(error)) from error


def require_roles(identity: IdentityContext, *roles: str) -> IdentityContext:
    if not identity.has_any_role(*roles):
        raise AuthError(403, f"One of these roles is required: {', '.join(roles)}")
    return identity
