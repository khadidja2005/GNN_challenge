import Link from 'next/link';
import { Upload, GitBranch, GitPullRequest, Github, Trophy } from 'lucide-react';

export default function SubmitPage() {
  return (
    <div className="min-h-screen py-12 px-4">
      <div className="max-w-3xl mx-auto">
        {/* Header */}
        <div className="mb-12 text-center">
          <div className="inline-flex items-center gap-2 text-purple-400 mb-4">
            <Upload className="w-5 h-5" />
            <span>Submit via Pull Request</span>
          </div>
          <h1 className="text-4xl font-bold mb-4">Get on the Leaderboard</h1>
          <p className="text-gray-400 text-lg">
            Submit your predictions and code by opening a pull request. Approved PRs appear on the leaderboard.
          </p>
        </div>

        {/* Steps */}
        <div className="space-y-6">
          <div className="p-5 rounded-xl border border-gray-800 bg-gray-900/40">
            <div className="flex items-center gap-3 mb-2 text-gray-200 font-semibold">
              <GitBranch className="w-5 h-5 text-emerald-400" />
              1) Fork & branch
            </div>
            <p className="text-sm text-gray-500">Fork the repo, create a feature branch, and generate your predictions with one of the submission scripts.</p>
          </div>

          <div className="p-5 rounded-xl border border-gray-800 bg-gray-900/40">
            <div className="flex items-center gap-3 mb-2 text-gray-200 font-semibold">
              <Upload className="w-5 h-5 text-blue-400" />
              2) Add predictions
            </div>
            <p className="text-sm text-gray-500">
              Place <code className="text-purple-300">predictions.csv</code> under <code className="text-purple-300">submissions/&lt;your_name&gt;/</code>. Include your training script (or the modified submission script) in the same folder.
            </p>
          </div>

          <div className="p-5 rounded-xl border border-gray-800 bg-gray-900/40">
            <div className="flex items-center gap-3 mb-2 text-gray-200 font-semibold">
              <GitPullRequest className="w-5 h-5 text-amber-400" />
              3) Open a Pull Request
            </div>
            <p className="text-sm text-gray-500">
              Title it "Submission: &lt;your model name&gt;". Describe your model (GCN/GAT/GraphSAGE/other), parameters (&lt;=100k), and training time (&lt;5 min CPU).
            </p>
          </div>

          <div className="p-5 rounded-xl border border-gray-800 bg-gray-900/40">
            <div className="flex items-center gap-3 mb-2 text-gray-200 font-semibold">
              <Trophy className="w-5 h-5 text-yellow-400" />
              4) Review & leaderboard
            </div>
            <p className="text-sm text-gray-500">
              Once the PR is reviewed and merged, your entry and scores will appear on the leaderboard.
            </p>
          </div>
        </div>

        {/* Helpful links */}
        <div className="mt-10 grid sm:grid-cols-2 gap-4">
          <a
            href="https://github.com/khadidja2005/GNN_challenge"
            target="_blank"
            rel="noopener noreferrer"
            className="p-4 rounded-xl border border-gray-800 bg-gray-900/40 hover:border-purple-500 transition-colors flex items-center gap-3"
          >
            <Github className="w-5 h-5 text-purple-400" />
            <div>
              <p className="text-gray-200 font-semibold">Repository</p>
              <p className="text-xs text-gray-500">Fork and open your PR here</p>
            </div>
          </a>
          <Link
            href="/docs"
            className="p-4 rounded-xl border border-gray-800 bg-gray-900/40 hover:border-emerald-500 transition-colors flex items-center gap-3"
          >
            <Upload className="w-5 h-5 text-emerald-400" />
            <div>
              <p className="text-gray-200 font-semibold">Submission format</p>
              <p className="text-xs text-gray-500">See CSV requirements and rules</p>
            </div>
          </Link>
        </div>
      </div>
    </div>
  );
}
