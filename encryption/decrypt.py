#!/usr/bin/env python3
"""
ENZYMES-Hard Challenge: Decryption Script (CI Only)

Decrypts encrypted submission files using the private RSA key.
This script is used ONLY by the CI system with the secret private key.

Usage (CI only):
    python encryption/decrypt.py submissions/team.enc /path/to/private_key.pem output.csv

Note: The private key is stored in GitHub Secrets and never exposed publicly.
"""

import sys
import argparse
from pathlib import Path

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
except ImportError:
    print("❌ Error: cryptography package not installed.")
    print("   Run: pip install cryptography")
    sys.exit(1)


def load_private_key(key_path: str, password: bytes = None):
    """Load RSA private key from PEM file."""
    with open(key_path, 'rb') as f:
        private_key = serialization.load_pem_private_key(
            f.read(),
            password=password,
            backend=default_backend()
        )
    return private_key


def decrypt_file(input_path: str, private_key_path: str, output_path: str):
    """
    Decrypt a file encrypted with hybrid encryption (RSA + AES).
    
    Format: [4 bytes: key_len][encrypted_key][12 bytes: iv][16 bytes: tag][ciphertext]
    """
    # Load private key
    private_key = load_private_key(private_key_path)
    
    # Read encrypted file
    with open(input_path, 'rb') as f:
        data = f.read()
    
    # Parse the encrypted data
    offset = 0
    
    # Read encrypted key length
    encrypted_key_len = int.from_bytes(data[offset:offset+4], 'big')
    offset += 4
    
    # Read encrypted AES key
    encrypted_aes_key = data[offset:offset+encrypted_key_len]
    offset += encrypted_key_len
    
    # Read IV (12 bytes for GCM)
    iv = data[offset:offset+12]
    offset += 12
    
    # Read tag (16 bytes for GCM)
    tag = data[offset:offset+16]
    offset += 16
    
    # Read ciphertext
    ciphertext = data[offset:]
    
    # Decrypt AES key with RSA private key
    aes_key = private_key.decrypt(
        encrypted_aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    
    # Decrypt data with AES-GCM
    cipher = Cipher(algorithms.AES(aes_key), modes.GCM(iv, tag), backend=default_backend())
    decryptor = cipher.decryptor()
    plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    
    # Write to output file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(plaintext)
    
    return len(plaintext)


def main():
    parser = argparse.ArgumentParser(
        description='Decrypt submission for ENZYMES-Hard Challenge (CI only)'
    )
    parser.add_argument('input', help='Path to encrypted submission file (.enc)')
    parser.add_argument('private_key', help='Path to private key PEM file')
    parser.add_argument('output', help='Output path for decrypted CSV')
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input).exists():
        print(f"❌ Error: Encrypted file not found: {args.input}")
        sys.exit(1)
    
    if not Path(args.private_key).exists():
        print(f"❌ Error: Private key not found: {args.private_key}")
        sys.exit(1)
    
    try:
        size = decrypt_file(args.input, args.private_key, args.output)
        print(f"✅ Decryption successful: {size:,} bytes written to {args.output}")
    except Exception as e:
        print(f"❌ Decryption failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
