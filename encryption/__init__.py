"""
Encryption utilities for secure submission handling.

This module provides RSA-4096 + AES-256-GCM hybrid encryption for
protecting participant predictions during the submission process.

Files:
- encrypt.py: Encrypt predictions.csv with the public key (for participants)
- decrypt.py: Decrypt .enc files with the private key (for CI only)
- generate_keys.py: Generate a new RSA-4096 key pair (for organizers)
- public_key.pem: The public key (committed to repo)

Usage (for participants):
    python encryption/encrypt.py predictions.csv encryption/public_key.pem submissions/team.enc

The private key (private_key.pem) should NEVER be committed to the repo.
It must be stored as a GitHub Secret (RSA_PRIVATE_KEY) for CI decryption.
"""
