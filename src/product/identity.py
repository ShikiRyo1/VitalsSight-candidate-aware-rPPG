from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Iterable, Mapping


DEFAULT_ORGANIZATION_ID = "local-research"
DEFAULT_USER_ID = "local-operator"
DEFAULT_ORGANIZATION_NAME = "Local research workspace"

ROLE_PARTICIPANT = "participant"
ROLE_OPERATOR = "operator"
ROLE_REVIEWER = "reviewer"
ROLE_RESEARCHER = "researcher"
ROLE_AUDITOR = "auditor"
ROLE_ORG_ADMIN = "org-admin"
ROLE_SERVICE = "service"

ALL_ROLES = frozenset(
    {
        ROLE_PARTICIPANT,
        ROLE_OPERATOR,
        ROLE_REVIEWER,
        ROLE_RESEARCHER,
        ROLE_AUDITOR,
        ROLE_ORG_ADMIN,
        ROLE_SERVICE,
    }
)

LOCAL_ADMIN_ROLES = frozenset(
    {
        ROLE_OPERATOR,
        ROLE_REVIEWER,
        ROLE_RESEARCHER,
        ROLE_AUDITOR,
        ROLE_ORG_ADMIN,
    }
)


def normalize_identifier(value: object, *, fallback: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text).strip("-._")
    return text[:96] or fallback


def _flatten_roles(value: object) -> set[str]:
    if isinstance(value, str):
        values: Iterable[object] = re.split(r"[\s,]+", value)
    elif isinstance(value, (list, tuple, set, frozenset)):
        values = value
    else:
        return set()
    return {
        normalize_identifier(item, fallback="")
        for item in values
        if normalize_identifier(item, fallback="") in ALL_ROLES
    }


def roles_from_claims(claims: Mapping[str, Any], *, client_id: str = "") -> frozenset[str]:
    roles = _flatten_roles(claims.get("roles"))
    realm_access = claims.get("realm_access")
    if isinstance(realm_access, Mapping):
        roles.update(_flatten_roles(realm_access.get("roles")))
    resource_access = claims.get("resource_access")
    if isinstance(resource_access, Mapping):
        selected = resource_access.get(client_id) if client_id else None
        if isinstance(selected, Mapping):
            roles.update(_flatten_roles(selected.get("roles")))
        elif not client_id:
            for value in resource_access.values():
                if isinstance(value, Mapping):
                    roles.update(_flatten_roles(value.get("roles")))
    return frozenset(roles)


def organizations_from_claims(claims: Mapping[str, Any]) -> frozenset[str]:
    organizations: set[str] = set()
    explicit = claims.get("organization_id") or claims.get("org_id") or claims.get("tenant_id")
    if explicit:
        organizations.add(normalize_identifier(explicit, fallback=DEFAULT_ORGANIZATION_ID))
    organization_claim = claims.get("organization")
    if isinstance(organization_claim, str):
        organizations.add(normalize_identifier(organization_claim, fallback=DEFAULT_ORGANIZATION_ID))
    elif isinstance(organization_claim, Mapping):
        organizations.update(
            normalize_identifier(key, fallback=DEFAULT_ORGANIZATION_ID)
            for key in organization_claim.keys()
        )
    elif isinstance(organization_claim, (list, tuple, set, frozenset)):
        organizations.update(
            normalize_identifier(item, fallback=DEFAULT_ORGANIZATION_ID)
            for item in organization_claim
        )
    return frozenset(organizations)


@dataclass(frozen=True)
class IdentityContext:
    user_id: str
    subject: str
    email: str
    display_name: str
    organization_id: str
    roles: frozenset[str] = field(default_factory=frozenset)
    auth_mode: str = "disabled"
    token_id: str = ""
    participant_id: str = ""

    @property
    def actor(self) -> str:
        return self.display_name or self.email or self.user_id

    @property
    def primary_role(self) -> str:
        priority = (
            ROLE_ORG_ADMIN,
            ROLE_REVIEWER,
            ROLE_OPERATOR,
            ROLE_RESEARCHER,
            ROLE_AUDITOR,
            ROLE_PARTICIPANT,
            ROLE_SERVICE,
        )
        return next((role for role in priority if role in self.roles), "")

    def has_any_role(self, *roles: str) -> bool:
        return bool(self.roles.intersection(roles))

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "subject": self.subject,
            "email": self.email,
            "display_name": self.display_name,
            "organization_id": self.organization_id,
            "roles": sorted(self.roles),
            "auth_mode": self.auth_mode,
            "token_id": self.token_id,
            "participant_id": self.participant_id,
        }


def local_identity(
    *,
    organization_id: str = DEFAULT_ORGANIZATION_ID,
    user_id: str = DEFAULT_USER_ID,
    display_name: str = "Research operator",
    roles: Iterable[str] = LOCAL_ADMIN_ROLES,
) -> IdentityContext:
    normalized_user = normalize_identifier(user_id, fallback=DEFAULT_USER_ID)
    normalized_org = normalize_identifier(organization_id, fallback=DEFAULT_ORGANIZATION_ID)
    normalized_roles = frozenset(role for role in roles if role in ALL_ROLES)
    return IdentityContext(
        user_id=normalized_user,
        subject=f"local:{normalized_user}",
        email="",
        display_name=display_name.strip() or normalized_user,
        organization_id=normalized_org,
        roles=normalized_roles or LOCAL_ADMIN_ROLES,
        auth_mode="disabled",
    )


def identity_from_claims(
    claims: Mapping[str, Any],
    *,
    organization_id: str,
    auth_mode: str = "oidc",
    client_id: str = "",
) -> IdentityContext:
    subject = str(claims.get("sub") or "").strip()
    if not subject:
        raise ValueError("OIDC identity is missing the subject claim")
    email = str(claims.get("email") or "").strip()
    display_name = str(claims.get("name") or claims.get("preferred_username") or email or subject).strip()
    user_id = normalize_identifier(subject, fallback=DEFAULT_USER_ID)
    return IdentityContext(
        user_id=user_id,
        subject=subject,
        email=email,
        display_name=display_name,
        organization_id=normalize_identifier(organization_id, fallback=DEFAULT_ORGANIZATION_ID),
        roles=roles_from_claims(claims, client_id=client_id),
        auth_mode=auth_mode,
        token_id=str(claims.get("jti") or ""),
        participant_id=normalize_identifier(claims.get("participant_id"), fallback=""),
    )
