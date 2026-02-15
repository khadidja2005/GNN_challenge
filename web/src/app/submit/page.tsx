import Link from 'next/link';
import { Lock, Trophy, Shield, FileText, Terminal, GitPullRequest, Key } from 'lucide-react';

export default function SubmitPage() {
  return (
    <div className="min-h-screen py-12 px-4">
      <div className="max-w-3xl mx-auto">
        {/* Header */}
        <div className="mb-12 text-center">
          <div className="inline-flex items-center gap-2 text-purple-400 mb-4">
            <Lock className="w-5 h-5" />
            <span>Encrypted Submission</span>
          </div>
          <h1 className="text-4xl font-bold mb-4">Submit Your Solution</h1>
          <p className="text-gray-400 text-lg">
            Submissions are <strong className="text-purple-300">encrypted</strong> with RSA. Only the CI system can decrypt and evaluate your predictions.
          </p>
          <div className="mt-4 p-3 rounded-lg bg-amber-900/20 border border-amber-700/40 text-amber-300 text-sm">
            ⚠️ <strong>One submission per participant!</strong> Make sure your model is ready before submitting.
          </div>
        </div>

        {/* Security Notice */}
        <div className="mb-8 p-5 rounded-xl border border-emerald-800/50 bg-emerald-900/20">
          <div className="flex items-center gap-3 mb-3 text-emerald-400 font-semibold">
            <Shield className="w-5 h-5" />
            🔐 How Encryption Protects Your Work
          </div>
          <ul className="text-sm text-gray-400 space-y-1 list-disc list-inside">
            <li>Your predictions are encrypted with <strong>RSA-4096 + AES-256</strong> before upload</li>
            <li>Only the private key (held in CI secrets) can decrypt your submission</li>
            <li>Other participants <strong>cannot</strong> see your predictions — even in the PR</li>
            <li>The leaderboard shows only your team name, score, and rank</li>
          </ul>
        </div>

        {/* Steps */}
        <div className="space-y-6">
          <div className="p-5 rounded-xl border border-gray-800 bg-gray-900/40">
            <div className="flex items-center gap-3 mb-2 text-gray-200 font-semibold">
              <FileText className="w-5 h-5 text-blue-400" />
              1) Prepare your predictions
            </div>
            <p className="text-sm text-gray-500">
              Generate <code className="text-purple-300">predictions.csv</code> with 180 rows — columns: <code className="text-purple-300">graph_id</code>, <code className="text-purple-300">prediction</code> (1-6).
            </p>
            <pre className="mt-3 p-3 rounded bg-gray-800 text-xs text-gray-300 overflow-x-auto">
{`graph_id,prediction
0,3
1,2
2,5
...`}
            </pre>
          </div>

          <div className="p-5 rounded-xl border border-gray-800 bg-gray-900/40">
            <div className="flex items-center gap-3 mb-2 text-gray-200 font-semibold">
              <Key className="w-5 h-5 text-amber-400" />
              2) Install encryption dependencies
            </div>
            <pre className="mt-2 p-3 rounded bg-gray-800 text-xs text-gray-300 overflow-x-auto">
{`pip install cryptography`}
            </pre>
          </div>

          <div className="p-5 rounded-xl border border-gray-800 bg-gray-900/40">
            <div className="flex items-center gap-3 mb-2 text-gray-200 font-semibold">
              <Terminal className="w-5 h-5 text-emerald-400" />
              3) Encrypt your submission
            </div>
            <p className="text-sm text-gray-500 mb-2">
              Run the encryption script with your predictions and the public key:
            </p>
            <pre className="p-3 rounded bg-gray-800 text-xs text-gray-300 overflow-x-auto">
{`python encryption/encrypt.py \\
    predictions.csv \\
    encryption/public_key.pem \\
    submissions/yourteam.enc`}
            </pre>
            <p className="mt-2 text-xs text-gray-500">
              Replace <code className="text-purple-300">yourteam</code> with your team name (no spaces).
            </p>
          </div>

          <div className="p-5 rounded-xl border border-gray-800 bg-gray-900/40">
            <div className="flex items-center gap-3 mb-2 text-gray-200 font-semibold">
              <GitPullRequest className="w-5 h-5 text-purple-400" />
              4) Submit via Pull Request
            </div>
            <p className="text-sm text-gray-500">
              Open a PR adding <strong>only</strong> your encrypted file:
            </p>
            <ul className="mt-2 text-sm text-gray-400 space-y-1 list-disc list-inside">
              <li>Fork the repository (if not a collaborator)</li>
              <li>Add: <code className="text-purple-300">submissions/yourteam.enc</code></li>
              <li>PR title: <code className="text-purple-300">[Submission] YourTeamName</code></li>
              <li>CI automatically decrypts, validates, and evaluates</li>
            </ul>
          </div>

          <div className="p-5 rounded-xl border border-gray-800 bg-gray-900/40">
            <div className="flex items-center gap-3 mb-2 text-gray-200 font-semibold">
              <Trophy className="w-5 h-5 text-yellow-400" />
              5) Check Results
            </div>
            <p className="text-sm text-gray-500">
              CI will comment on your PR with your score. The leaderboard updates automatically.
              Your predictions remain encrypted — only the final score is public.
            </p>
          </div>
        </div>

        {/* Submission checklist */}
        <div className="mt-10 p-5 rounded-xl border border-gray-800 bg-gray-900/40">
          <h3 className="text-gray-200 font-semibold mb-3">📋 Pre-Submission Checklist</h3>
          <ul className="text-sm text-gray-400 space-y-2">
            <li className="flex items-center gap-2">
              <span className="text-emerald-400">✓</span> predictions.csv has exactly 180 rows
            </li>
            <li className="flex items-center gap-2">
              <span className="text-emerald-400">✓</span> Columns: graph_id (0-179), prediction (1-6)
            </li>
            <li className="flex items-center gap-2">
              <span className="text-emerald-400">✓</span> Model uses ≤100K parameters
            </li>
            <li className="flex items-center gap-2">
              <span className="text-emerald-400">✓</span> Training completes in ≤3h on CPU
            </li>
            <li className="flex items-center gap-2">
              <span className="text-emerald-400">✓</span> Encrypted with public_key.pem → .enc file
            </li>
            <li className="flex items-center gap-2">
              <span className="text-emerald-400">✓</span> Filename: submissions/yourteam.enc
            </li>
          </ul>
        </div>

        {/* Command summary */}
        <div className="mt-6 p-5 rounded-xl border border-purple-800/50 bg-purple-900/20">
          <h3 className="text-purple-300 font-semibold mb-3">⚡ Quick Commands</h3>
          <pre className="p-3 rounded bg-gray-900 text-xs text-gray-300 overflow-x-auto">
{`# Validate your predictions format
python scripts/validate_submission.py --predictions predictions.csv

# Encrypt for submission
python encryption/encrypt.py predictions.csv encryption/public_key.pem submissions/yourteam.enc

# Commit and push
git add submissions/yourteam.enc
git commit -m "[Submission] YourTeamName"
git push origin main`}
          </pre>
        </div>

        {/* Links */}
        <div className="mt-10 grid sm:grid-cols-2 gap-4">
          <Link
            href="/docs"
            className="p-4 rounded-xl border border-gray-800 bg-gray-900/40 hover:border-purple-500 transition-colors flex items-center gap-3"
          >
            <FileText className="w-5 h-5 text-purple-400" />
            <div>
              <p className="text-gray-200 font-semibold">Documentation</p>
              <p className="text-xs text-gray-500">CSV format & rules</p>
            </div>
          </Link>
          <Link
            href="/leaderboard"
            className="p-4 rounded-xl border border-gray-800 bg-gray-900/40 hover:border-emerald-500 transition-colors flex items-center gap-3"
          >
            <Trophy className="w-5 h-5 text-emerald-400" />
            <div>
              <p className="text-gray-200 font-semibold">Leaderboard</p>
              <p className="text-xs text-gray-500">See current rankings</p>
            </div>
          </Link>
        </div>
      </div>
    </div>
  );
}
