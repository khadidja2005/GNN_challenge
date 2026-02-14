import Link from 'next/link';
import { Upload, Mail, Lock, Trophy, Shield, FileText } from 'lucide-react';

export default function SubmitPage() {
  return (
    <div className="min-h-screen py-12 px-4">
      <div className="max-w-3xl mx-auto">
        {/* Header */}
        <div className="mb-12 text-center">
          <div className="inline-flex items-center gap-2 text-purple-400 mb-4">
            <Lock className="w-5 h-5" />
            <span>Private Submission</span>
          </div>
          <h1 className="text-4xl font-bold mb-4">Submit Your Solution</h1>
          <p className="text-gray-400 text-lg">
            Submissions are <strong className="text-purple-300">private</strong>. Only your final score and rank appear on the public leaderboard.
          </p>
          <div className="mt-4 p-3 rounded-lg bg-amber-900/20 border border-amber-700/40 text-amber-300 text-sm">
            ⚠️ <strong>One submission per participant!</strong> Make sure your model is ready before submitting.
          </div>
        </div>

        {/* Privacy Notice */}
        <div className="mb-8 p-5 rounded-xl border border-emerald-800/50 bg-emerald-900/20">
          <div className="flex items-center gap-3 mb-3 text-emerald-400 font-semibold">
            <Shield className="w-5 h-5" />
            Submission Privacy Guarantee
          </div>
          <ul className="text-sm text-gray-400 space-y-1 list-disc list-inside">
            <li>Your code and predictions are <strong>never</strong> made public</li>
            <li>Only your team name, score, and rank appear on the leaderboard</li>
            <li>Other participants cannot see your solution</li>
            <li>Solutions may be shared <strong>after</strong> competition ends (with your permission)</li>
          </ul>
        </div>

        {/* Steps */}
        <div className="space-y-6">
          <div className="p-5 rounded-xl border border-gray-800 bg-gray-900/40">
            <div className="flex items-center gap-3 mb-2 text-gray-200 font-semibold">
              <FileText className="w-5 h-5 text-blue-400" />
              1) Prepare your submission
            </div>
            <p className="text-sm text-gray-500">
              Create a folder with your <code className="text-purple-300">predictions.csv</code> (180 rows, columns: graph_id, prediction), 
              training script, and a brief <code className="text-purple-300">README.md</code> describing your model.
            </p>
          </div>

          <div className="p-5 rounded-xl border border-gray-800 bg-gray-900/40">
            <div className="flex items-center gap-3 mb-2 text-gray-200 font-semibold">
              <Upload className="w-5 h-5 text-emerald-400" />
              2) Zip your submission
            </div>
            <p className="text-sm text-gray-500">
              Zip the folder containing: <code className="text-purple-300">predictions.csv</code>, your code, and README.
              Name it <code className="text-purple-300">submission_yourname.zip</code>.
            </p>
          </div>

          <div className="p-5 rounded-xl border border-gray-800 bg-gray-900/40">
            <div className="flex items-center gap-3 mb-2 text-gray-200 font-semibold">
              <Mail className="w-5 h-5 text-amber-400" />
              3) Submit privately
            </div>
            <p className="text-sm text-gray-500">
              Send your submission via <strong>one</strong> of these private channels:
            </p>
            <ul className="mt-2 text-sm text-gray-400 space-y-1 list-disc list-inside">
              <li>Email to: <code className="text-purple-300">khadidja.benkermiche@ensia.edu.dz</code></li>
              <li>GitHub DM to <a href="https://github.com/khadidja2005" className="text-purple-400 hover:underline" target="_blank" rel="noopener noreferrer">@khadidja2005</a></li>
              <li>Share a <strong>private</strong> repo link (invite the organizer as collaborator)</li>
            </ul>
          </div>

          <div className="p-5 rounded-xl border border-gray-800 bg-gray-900/40">
            <div className="flex items-center gap-3 mb-2 text-gray-200 font-semibold">
              <Trophy className="w-5 h-5 text-yellow-400" />
              4) Evaluation & leaderboard
            </div>
            <p className="text-sm text-gray-500">
              The organizer will evaluate your submission on the hidden test set and add your score to the public leaderboard.
              You will receive confirmation once your entry is live.
            </p>
          </div>
        </div>

        {/* Submission checklist */}
        <div className="mt-10 p-5 rounded-xl border border-gray-800 bg-gray-900/40">
          <h3 className="text-gray-200 font-semibold mb-3">📋 Submission Checklist</h3>
          <ul className="text-sm text-gray-400 space-y-2">
            <li className="flex items-center gap-2">
              <span className="text-emerald-400">✓</span> predictions.csv (180 rows, graph_id + prediction columns)
            </li>
            <li className="flex items-center gap-2">
              <span className="text-emerald-400">✓</span> Training code (reproducible with seed)
            </li>
            <li className="flex items-center gap-2">
              <span className="text-emerald-400">✓</span> README with model description, params (≤100K), training time (≤3h)
            </li>
            <li className="flex items-center gap-2">
              <span className="text-emerald-400">✓</span> requirements.txt or environment file
            </li>
            <li className="flex items-center gap-2">
              <span className="text-gray-500">○</span> Model weights (optional but recommended)
            </li>
          </ul>
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
