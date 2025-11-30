'use client';

import { Brain, Zap, Target, Clock, Database, Award } from 'lucide-react';
import Link from 'next/link';

const features = [
  {
    icon: Database,
    title: 'Limited Data',
    description: 'Only 240 training graphs available. Learn to generalize from minimal examples.',
    color: 'text-purple-400',
  },
  {
    icon: Target,
    title: 'Class Imbalance',
    description: 'Validation set has skewed class distribution. Handle imbalance effectively.',
    color: 'text-emerald-400',
  },
  {
    icon: Zap,
    title: 'Missing Features',
    description: '10-15% of node features are missing. Build robust representations.',
    color: 'text-amber-400',
  },
  {
    icon: Clock,
    title: 'Time Constraint',
    description: 'Model must train in under 5 minutes on CPU. Efficiency matters.',
    color: 'text-blue-400',
  },
];

const stats = [
  { value: '600', label: 'Protein Graphs' },
  { value: '6', label: 'Enzyme Classes' },
  { value: '18', label: 'Node Features' },
  { value: '100K', label: 'Max Parameters' },
];

export default function Home() {
  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative overflow-hidden py-20 px-4">
        <div className="absolute inset-0 bg-gradient-to-b from-purple-900/20 to-transparent pointer-events-none" />
        
        <div className="max-w-6xl mx-auto text-center relative z-10">
          <div className="inline-flex items-center gap-2 bg-purple-500/10 border border-purple-500/30 rounded-full px-4 py-2 mb-6">
            <Brain className="w-4 h-4 text-purple-400" />
            <span className="text-sm text-purple-300">Graph Neural Network Challenge</span>
          </div>
          
          <h1 className="text-5xl md:text-7xl font-bold mb-6">
            <span className="gradient-text">ENZYMES-Hard</span>
          </h1>
          
          <p className="text-xl md:text-2xl text-gray-400 mb-4">
            Few-Shot Protein Function Classification
          </p>
          
          <p className="text-lg text-gray-500 max-w-2xl mx-auto mb-8">
            Classify enzyme proteins into 6 EC top-level classes with limited training data, 
            missing features, and strict model constraints. Can your GNN rise to the challenge?
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              href="/docs"
              className="px-8 py-4 bg-gradient-to-r from-purple-600 to-indigo-600 rounded-lg font-semibold 
                       hover:from-purple-500 hover:to-indigo-500 transition-all glow"
            >
              Get Started
            </Link>
            <Link
              href="/submit"
              className="px-8 py-4 bg-gray-800 border border-gray-700 rounded-lg font-semibold 
                       hover:bg-gray-700 hover:border-gray-600 transition-all"
            >
              Submit Solution
            </Link>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 px-4 border-y border-gray-800">
        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, idx) => (
              <div key={idx} className="text-center">
                <div className="text-4xl md:text-5xl font-bold gradient-text mb-2">
                  {stat.value}
                </div>
                <div className="text-gray-500">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Challenges Section */}
      <section className="py-20 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">The Challenge</h2>
            <p className="text-gray-500 max-w-2xl mx-auto">
              This isn't your typical graph classification task. We've added multiple difficulty 
              modifiers to test your GNN skills.
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 gap-6">
            {features.map((feature, idx) => (
              <div
                key={idx}
                className="p-6 rounded-xl bg-gray-900/50 border border-gray-800 card-hover"
              >
                <feature.icon className={`w-10 h-10 ${feature.color} mb-4`} />
                <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                <p className="text-gray-400">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Evaluation Section */}
      <section className="py-20 px-4 bg-gray-900/30">
        <div className="max-w-6xl mx-auto">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div>
              <h2 className="text-3xl md:text-4xl font-bold mb-6">
                <Award className="inline w-10 h-10 text-amber-400 mr-3" />
                Evaluation
              </h2>
              <p className="text-gray-400 mb-6">
                Your solution will be evaluated on the held-out test set of 180 graphs. 
                The primary metric is <span className="text-emerald-400 font-semibold">Macro F1-Score</span>, 
                which equally weights all 6 classes regardless of frequency.
              </p>
              <div className="space-y-4">
                <div className="flex items-center gap-4">
                  <div className="w-3 h-3 rounded-full bg-emerald-400" />
                  <span className="text-gray-300">Primary: Macro F1-Score</span>
                </div>
                <div className="flex items-center gap-4">
                  <div className="w-3 h-3 rounded-full bg-blue-400" />
                  <span className="text-gray-300">Secondary: Overall Accuracy</span>
                </div>
                <div className="flex items-center gap-4">
                  <div className="w-3 h-3 rounded-full bg-purple-400" />
                  <span className="text-gray-300">Bonus: Per-class breakdown</span>
                </div>
              </div>
            </div>
            
            <div className="p-6 rounded-xl bg-gray-800/50 border border-gray-700 code-block">
              <div className="text-sm text-gray-500 mb-4"># Baseline Performance</div>
              <div className="space-y-3 font-mono text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Model:</span>
                  <span className="text-purple-400">GIN (3 layers)</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Parameters:</span>
                  <span className="text-emerald-400">29,350</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Training Time:</span>
                  <span className="text-blue-400">~5 seconds</span>
                </div>
                <div className="border-t border-gray-700 my-2" />
                <div className="flex justify-between">
                  <span className="text-gray-400">Macro F1:</span>
                  <span className="text-amber-400 font-bold">~0.29</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Accuracy:</span>
                  <span className="text-amber-400">~32%</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl md:text-4xl font-bold mb-6">Ready to Begin?</h2>
          <p className="text-gray-400 mb-8">
            Download the dataset, build your GNN, and submit your predictions to see how you rank.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              href="/docs"
              className="px-8 py-4 bg-gradient-to-r from-purple-600 to-indigo-600 rounded-lg font-semibold 
                       hover:from-purple-500 hover:to-indigo-500 transition-all"
            >
              Read Documentation
            </Link>
            <Link
              href="/leaderboard"
              className="px-8 py-4 bg-gray-800 border border-gray-700 rounded-lg font-semibold 
                       hover:bg-gray-700 transition-all"
            >
              View Leaderboard
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 px-4 border-t border-gray-800">
        <div className="max-w-6xl mx-auto text-center text-gray-500 text-sm">
          <p>ENZYMES-Hard Challenge • Built with PyTorch Geometric</p>
        </div>
      </footer>
    </div>
  );
}
