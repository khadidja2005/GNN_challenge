'use client';

import { Trophy, Medal, TrendingUp, Clock, User, Cpu } from 'lucide-react';
import leaderboardData from './data.json';

interface LeaderboardEntry {
  rank: number;
  name: string;
  macroF1: number;
  accuracy: number;
  params: string;
  trainTime: string;
  submittedAt: string;
  isBaseline?: boolean;
}

const leaderboardEntries = leaderboardData as LeaderboardEntry[];

function getRankIcon(rank: number) {
  switch (rank) {
    case 1:
      return <Trophy className="w-6 h-6 text-yellow-400" />;
    case 2:
      return <Medal className="w-6 h-6 text-gray-300" />;
    case 3:
      return <Medal className="w-6 h-6 text-amber-600" />;
    default:
      return <span className="text-gray-500 font-mono w-6 text-center">{rank}</span>;
  }
}

function getRowStyle(entry: LeaderboardEntry) {
  if (entry.isBaseline) {
    return 'bg-amber-900/10 border-amber-800/30';
  }
  if (entry.rank === 1) {
    return 'bg-yellow-900/10 border-yellow-800/30';
  }
  if (entry.rank <= 3) {
    return 'bg-purple-900/10 border-purple-800/30';
  }
  return 'bg-gray-900/30 border-gray-800';
}

export default function LeaderboardPage() {
  const topScore = leaderboardEntries.reduce((max, entry) => Math.max(max, entry.macroF1), 0) || 1;
  const baselineScore =
    leaderboardEntries.find((entry) => entry.isBaseline)?.macroF1 ?? 0.29;

  return (
    <div className="min-h-screen py-12 px-4">
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="mb-12 text-center">
          <div className="inline-flex items-center gap-2 text-purple-400 mb-4">
            <Trophy className="w-5 h-5" />
            <span>Leaderboard</span>
          </div>
          <h1 className="text-4xl font-bold mb-4">Top Submissions</h1>
          <p className="text-gray-400 text-lg">
            See how participants are performing on the ENZYMES-Hard challenge.
          </p>
        </div>

        {/* Stats Cards */}
        <div className="grid md:grid-cols-4 gap-4 mb-8">
          <div className="p-4 bg-gradient-to-br from-yellow-900/30 to-yellow-900/10 border border-yellow-800/30 rounded-xl text-center">
            <Trophy className="w-8 h-8 mx-auto mb-2 text-yellow-400" />
            <div className="text-2xl font-bold text-yellow-400">{topScore.toFixed(3)}</div>
            <div className="text-xs text-gray-500">Best Macro F1</div>
          </div>
          <div className="p-4 bg-gradient-to-br from-purple-900/30 to-purple-900/10 border border-purple-800/30 rounded-xl text-center">
            <User className="w-8 h-8 mx-auto mb-2 text-purple-400" />
            <div className="text-2xl font-bold text-purple-400">{leaderboardEntries.filter(e => !e.isBaseline).length}</div>
            <div className="text-xs text-gray-500">Participants</div>
          </div>
          <div className="p-4 bg-gradient-to-br from-emerald-900/30 to-emerald-900/10 border border-emerald-800/30 rounded-xl text-center">
            <TrendingUp className="w-8 h-8 mx-auto mb-2 text-emerald-400" />
            <div className="text-2xl font-bold text-emerald-400">
              +{((topScore - baselineScore) / baselineScore * 100).toFixed(0)}%
            </div>
            <div className="text-xs text-gray-500">vs Baseline</div>
          </div>
          <div className="p-4 bg-gradient-to-br from-blue-900/30 to-blue-900/10 border border-blue-800/30 rounded-xl text-center">
            <Cpu className="w-8 h-8 mx-auto mb-2 text-blue-400" />
            <div className="text-2xl font-bold text-blue-400">100K</div>
            <div className="text-xs text-gray-500">Max Params</div>
          </div>
        </div>

        {/* Leaderboard Table */}
        <div className="rounded-xl border border-gray-800 overflow-hidden">
          {/* Header */}
          <div className="grid grid-cols-12 gap-4 p-4 bg-gray-900/70 text-sm font-medium text-gray-400 border-b border-gray-800">
            <div className="col-span-1 text-center">#</div>
            <div className="col-span-3">Participant</div>
            <div className="col-span-2 text-center">Macro F1</div>
            <div className="col-span-2 text-center">Accuracy</div>
            <div className="col-span-2 text-center">Parameters</div>
            <div className="col-span-2 text-center">Train Time</div>
          </div>

          {/* Rows */}
          {leaderboardEntries.map((entry, idx) => (
            <div
              key={idx}
              className={`grid grid-cols-12 gap-4 p-4 border-b last:border-b-0 items-center transition-colors hover:bg-gray-800/30 ${getRowStyle(entry)}`}
            >
              <div className="col-span-1 flex justify-center">
                {getRankIcon(entry.rank)}
              </div>
              <div className="col-span-3">
                <div className="font-semibold text-gray-200 flex items-center gap-2">
                  {entry.name}
                  {entry.isBaseline && (
                    <span className="text-xs px-2 py-0.5 bg-amber-900/30 text-amber-400 rounded-full">
                      Baseline
                    </span>
                  )}
                </div>
                <div className="text-xs text-gray-500 flex items-center gap-1 mt-1">
                  <Clock className="w-3 h-3" />
                  {entry.submittedAt}
                </div>
              </div>
              <div className="col-span-2 text-center">
                <div className="font-mono font-bold text-lg text-emerald-400">
                  {entry.macroF1.toFixed(3)}
                </div>
                {/* Score bar */}
                <div className="w-full bg-gray-700 rounded-full h-1.5 mt-2">
                  <div 
                    className={`h-1.5 rounded-full transition-all ${entry.isBaseline ? 'bg-amber-500' : 'bg-emerald-500'}`}
                    style={{ width: `${(entry.macroF1 / topScore) * 100}%` }}
                  />
                </div>
              </div>
              <div className="col-span-2 text-center font-mono text-gray-300">
                {(entry.accuracy * 100).toFixed(1)}%
              </div>
              <div className="col-span-2 text-center font-mono text-gray-400 text-sm">
                {entry.params}
              </div>
              <div className="col-span-2 text-center font-mono text-gray-400 text-sm">
                {entry.trainTime}
              </div>
            </div>
          ))}
        </div>

        {/* Info Section */}
        <div className="mt-8 grid md:grid-cols-2 gap-4">
          <div className="p-4 bg-gray-900/30 border border-gray-800 rounded-xl">
            <h4 className="font-semibold text-gray-200 mb-2">📊 Scoring</h4>
            <p className="text-sm text-gray-500">
              Submissions are ranked by Macro F1-Score on the held-out test set of 180 graphs. 
              This metric equally weights all 6 enzyme classes.
            </p>
          </div>
          <div className="p-4 bg-gray-900/30 border border-gray-800 rounded-xl">
            <h4 className="font-semibold text-gray-200 mb-2">⏱️ Constraints</h4>
            <p className="text-sm text-gray-500">
              Models must have ≤100K parameters and train in under 5 minutes on CPU. 
              Submissions violating these rules are disqualified.
            </p>
          </div>
        </div>

        {/* CTA */}
        <div className="mt-12 text-center">
          <p className="text-gray-400 mb-4">Think you can do better?</p>
          <a
            href="/submit"
            className="inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-purple-600 to-indigo-600 rounded-lg font-semibold 
                     hover:from-purple-500 hover:to-indigo-500 transition-all"
          >
            Submit Your Solution
          </a>
        </div>
      </div>
    </div>
  );
}
