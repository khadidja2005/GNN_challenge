#!/usr/bin/env python3
"""
ENZYMES-Hard Challenge: Key Generation Script (Organizer Only)

Generates RSA key pair for the encrypted submission system.
- public_key.pem: Distributed to participants (commit to repo)
- private_key.pem: Keep SECRET, add to GitHub Secrets as RSA_PRIVATE_KEY

Usage:
    python encryption/generate_keys.py

After running:
    1. Commit public_key.pem to the repository
    2. Add private_key.pem contents to GitHub Secrets as RSA_PRIVATE_KEY
    3. DELETE the local private_key.pem file!
"""

from pathlib import Path

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.backends import default_backend
except ImportError:
    print("❌ Error: cryptography package not installed.")
    print("   Run: pip install cryptography")
    exit(1)


def generate_key_pair(output_dir: str = "encryption"):
    """Generate RSA-4096 key pair."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("🔑 Generating RSA-4096 key pair...")
    
    # Generate private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=4096,
        backend=default_backend()
    )
    
    # Get public key
    public_key = private_key.public_key()
    
    # Serialize private key (PEM format, no password)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    
    # Serialize public key (PEM format)
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    
    # Write keys to files
    private_key_path = output_path / "private_key.pem"
    public_key_path = output_path / "public_key.pem"
    
    with open(private_key_path, 'wb') as f:
        f.write(private_pem)
    
    with open(public_key_path, 'wb') as f:
        f.write(public_pem)
    
    print(f"✅ Keys generated successfully!")
    print()
    print(f"📁 Files created:")
    print(f"   Public key:  {public_key_path}")
    print(f"   Private key: {private_key_path}")
    print()
    print("⚠️  IMPORTANT NEXT STEPS:")
    print("   1. Commit public_key.pem to the repository")
    print("   2. Copy the ENTIRE contents of private_key.pem")
    print("   3. Go to GitHub → Settings → Secrets → Actions")
    print("   4. Create secret named: RSA_PRIVATE_KEY")
    print("   5. Paste the private key contents")
    print("   6. DELETE private_key.pem from your local machine!")
    print()
    print("🔐 The private key should NEVER be committed to the repository!")


if __name__ == '__main__':
    generate_key_pair()
