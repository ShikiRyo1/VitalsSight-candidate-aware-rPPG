from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
import pytest

from src.product.auth import AuthError, AuthSettings, IdentityResolver, require_roles


ISSUER = "https://identity.example.test/realms/vitalssight"
AUDIENCE = "vitalssight-api"
SECRET = "controlled-trial-test-secret-that-is-not-used-in-production"


def _settings(**overrides: object) -> AuthSettings:
    values = {
        "mode": "required",
        "issuer": ISSUER,
        "audience": AUDIENCE,
        "client_id": "vitalssight",
        "shared_secret": SECRET,
        "algorithms": ("HS256",),
        "leeway_seconds": 0,
    }
    values.update(overrides)
    return AuthSettings(**values)


def _claims(**overrides: object) -> dict[str, object]:
    now = datetime.now(UTC)
    values: dict[str, object] = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": "user-123",
        "email": "reviewer@example.test",
        "name": "Trial reviewer",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=5)).timestamp()),
        "organization": {"hospital-a": {"id": "org-a"}},
        "realm_access": {"roles": ["reviewer", "auditor"]},
    }
    values.update(overrides)
    return values


def _token(**overrides: object) -> str:
    return jwt.encode(_claims(**overrides), SECRET, algorithm="HS256")


def test_disabled_auth_returns_explicit_local_identity() -> None:
    identity = IdentityResolver(AuthSettings(mode="disabled")).resolve(None)

    assert identity.auth_mode == "disabled"
    assert identity.organization_id == "local-research"
    assert identity.has_any_role("operator", "org-admin")


def test_required_auth_accepts_verified_tenant_and_roles() -> None:
    identity = IdentityResolver(_settings()).resolve(f"Bearer {_token()}")

    assert identity.auth_mode == "oidc"
    assert identity.organization_id == "hospital-a"
    assert identity.email == "reviewer@example.test"
    assert identity.roles == frozenset({"reviewer", "auditor"})
    assert require_roles(identity, "reviewer") is identity


def test_required_auth_rejects_missing_expired_and_wrong_audience_tokens() -> None:
    resolver = IdentityResolver(_settings())
    with pytest.raises(AuthError, match="bearer token") as missing:
        resolver.resolve(None)
    assert missing.value.status_code == 401

    expired_at = int((datetime.now(UTC) - timedelta(minutes=1)).timestamp())
    with pytest.raises(AuthError, match="expired"):
        resolver.resolve(f"Bearer {_token(exp=expired_at)}")

    with pytest.raises(AuthError, match="audience"):
        resolver.resolve(f"Bearer {_token(aud='another-api')}")


def test_multiple_organizations_require_an_explicit_authorized_context() -> None:
    resolver = IdentityResolver(_settings())
    token = _token(organization={"hospital-a": {}, "hospital-b": {}})

    with pytest.raises(AuthError, match="Select one organization"):
        resolver.resolve(f"Bearer {token}")
    selected = resolver.resolve(
        f"Bearer {token}",
        headers={"X-VitalsSight-Organization": "hospital-b"},
    )
    assert selected.organization_id == "hospital-b"
    with pytest.raises(AuthError, match="not present"):
        resolver.resolve(
            f"Bearer {token}",
            headers={"X-VitalsSight-Organization": "hospital-c"},
        )


def test_role_guard_rejects_insufficient_role() -> None:
    identity = IdentityResolver(_settings()).resolve(f"Bearer {_token()}")
    with pytest.raises(AuthError, match="org-admin") as error:
        require_roles(identity, "org-admin")
    assert error.value.status_code == 403
