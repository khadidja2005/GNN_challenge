#!/usr/bin/env python3
"""
ENZYMES-Hard Challenge: Encryption Script

Encrypts your predictions CSV using the competition's public RSA key.
The encrypted file can ONLY be decrypted by the competition organizers.

Usage:
    python encryption/encrypt.py predictions.csv encryption/public_key.pem submissions/your_team.enc

Requirements:
    pip install cryptography
"""

import sys
import base64
import argparse
from pathlib import Path

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    import os
except ImportError:
    print("❌ Error: cryptography package not installed.")
    print("   Run: pip install cryptography")
    sys.exit(1)


def load_public_key(key_path: str):
    """Load RSA public key from PEM file."""
    with open(key_path, 'rb') as f:
        public_key = serialization.load_pem_public_key(f.read(), backend=default_backend())
    return public_key


def encrypt_file(input_path: str, public_key_path: str, output_path: str):
    """
    Encrypt a file using hybrid encryption (RSA + AES).
    
    - Generate random AES key
    - Encrypt file content with AES-256-GCM
    - Encrypt AES key with RSA public key
    - Output: RSA-encrypted AES key + IV + tag + ciphertext
    """
    # Load public key
    public_key = load_public_key(public_key_path)
    
    # Read input file
    with open(input_path, 'rb') as f:
        plaintext = f.read()
    
    # Generate random AES key and IV
    aes_key = os.urandom(32)  # AES-256
    iv = os.urandom(12)       # GCM recommended IV size
    
    # Encrypt data with AES-GCM
    cipher = Cipher(algorithms.AES(aes_key), modes.GCM(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    tag = encryptor.tag
    
    # Encrypt AES key with RSA public key
    encrypted_aes_key = public_key.encrypt(
        aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    
    # Pack everything together: 
    # [4 bytes: encrypted_key_len][encrypted_key][12 bytes: iv][16 bytes: tag][ciphertext]
    encrypted_key_len = len(encrypted_aes_key).to_bytes(4, 'big')
    
    output_data = encrypted_key_len + encrypted_aes_key + iv + tag + ciphertext
    
    # Write to output file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(output_data)
    
    return len(plaintext), len(output_data)


def main():
    parser = argparse.ArgumentParser(
        description='Encrypt predictions for ENZYMES-Hard Challenge submission'
    )
    parser.add_argument('input', help='Path to predictions CSV file')
    parser.add_argument('public_key', help='Path to public key PEM file')
    parser.add_argument('output', help='Output path for encrypted file (.enc)')
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input).exists():
        print(f"❌ Error: Input file not found: {args.input}")
        sys.exit(1)
    
    if not Path(args.public_key).exists():
        print(f"❌ Error: Public key not found: {args.public_key}")
        sys.exit(1)
    
    print(f"🔐 Encrypting {args.input}...")
    
    try:
        original_size, encrypted_size = encrypt_file(args.input, args.public_key, args.output)
        print(f"✅ Encryption successful!")
        print(f"   Original size:  {original_size:,} bytes")
        print(f"   Encrypted size: {encrypted_size:,} bytes")
        print(f"   Output file:    {args.output}")
        print()
        print("📤 Next steps:")
        print(f"   1. Create a PR to the competition repository")
        print(f"   2. Add your encrypted file to submissions/")
        print(f"   3. Push and wait for automated evaluation (2-5 min)")
    except Exception as e:
        print(f"❌ Encryption failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
