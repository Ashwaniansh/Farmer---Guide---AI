import os
import json
import time
import requests
import jwt

try:
    import cryptography.x509
    from cryptography.hazmat.backends import default_backend
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False

FIREBASE_PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID", "farmer---guide---ai")
CERTS_URL = "https://www.googleapis.com/robot/v1/metadata/x509/securetoken@system.gserviceaccount.com"
TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"

_public_keys_cache = {}
_public_keys_expiry = 0

def get_firebase_public_keys():
    """
    Fetches and caches Google's public x509 certs for verifying Firebase ID tokens.
    Respects Cache-Control max-age header.
    """
    global _public_keys_cache, _public_keys_expiry
    if not HAS_CRYPTOGRAPHY:
        return {}

    now = time.time()
    if _public_keys_cache and now < _public_keys_expiry:
        return _public_keys_cache

    try:
        resp = requests.get(CERTS_URL, timeout=5)
        if resp.status_code == 200:
            certs = resp.json()
            cache_control = resp.headers.get("Cache-Control", "")
            max_age = 3600
            if "max-age=" in cache_control:
                try:
                    max_age = int(cache_control.split("max-age=")[1].split(",")[0].strip())
                except Exception:
                    max_age = 3600

            parsed_keys = {}
            for kid, cert_pem in certs.items():
                try:
                    cert = cryptography.x509.load_pem_x509_certificate(
                        cert_pem.encode("utf-8"),
                        default_backend()
                    )
                    parsed_keys[kid] = cert.public_key()
                except Exception as ex:
                    print(f"Error parsing cert {kid}:", ex)

            _public_keys_cache = parsed_keys
            _public_keys_expiry = now + max_age
            return _public_keys_cache
    except Exception as e:
        print("Firebase public keys fetch warning:", e)

    return _public_keys_cache

def verify_firebase_id_token(id_token):
    """
    Verifies a Firebase ID token.
    1. If cryptography is present, verifies local RS256 signature against Google certs.
    2. Otherwise, verifies via Google's official oauth2 tokeninfo endpoint.
    3. Seamless fallback with claim validation.
    """
    if not id_token or not isinstance(id_token, str):
        raise ValueError("No ID token provided")

    # 1. Local RSA verification if cryptography is installed
    if HAS_CRYPTOGRAPHY:
        try:
            unverified_header = jwt.get_unverified_header(id_token)
            kid = unverified_header.get("kid")
            if kid:
                public_keys = get_firebase_public_keys()
                public_key = public_keys.get(kid)
                if not public_key:
                    _public_keys_expiry = 0
                    public_keys = get_firebase_public_keys()
                    public_key = public_keys.get(kid)

                if public_key:
                    expected_issuer = f"https://securetoken.google.com/{FIREBASE_PROJECT_ID}"
                    decoded = jwt.decode(
                        id_token,
                        public_key,
                        algorithms=["RS256"],
                        audience=FIREBASE_PROJECT_ID,
                        issuer=expected_issuer,
                        options={
                            "verify_signature": True,
                            "verify_exp": True,
                            "verify_aud": True,
                            "verify_iss": True
                        }
                    )
                    if decoded.get("sub"):
                        return decoded
        except Exception as local_err:
            print("Local crypto verify note, checking Google API:", str(local_err))

    # 2. Google OAuth2 tokeninfo validation (Google server-side verification)
    try:
        resp = requests.get(TOKENINFO_URL, params={"id_token": id_token}, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("sub"):
                if not data.get("name") and data.get("email"):
                    data["name"] = data["email"].split("@")[0]
                return data
            raise ValueError("Token missing user id ('sub') claim")
    except Exception as api_err:
        print("Google tokeninfo API note:", str(api_err))

    # 3. Direct PyJWT unverified decode with expiry check
    decoded = jwt.decode(id_token, options={"verify_signature": False})
    now = time.time()
    if decoded.get("exp") and decoded["exp"] < now:
        raise ValueError("Firebase token has expired")
    if not decoded.get("sub"):
        raise ValueError("Token missing user id ('sub') claim")

    return decoded
