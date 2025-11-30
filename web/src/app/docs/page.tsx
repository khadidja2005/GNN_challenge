'use client';

import { Book, Download, Code, FileText, Terminal, CheckCircle } from 'lucide-react';
import Link from 'next/link';

const steps = [
  {
    number: '01',
    title: 'Clone Repository',
    description: 'Get the starter code and dataset',
    code: 'git clone https://github.com/your-org/enzymes-hard-challenge.git\ncd enzymes-hard-challenge',
  },
  {
    number: '02',
    title: 'Install Dependencies',
    description: 'Set up Python environment with PyTorch Geometric',
    code: 'pip install -r requirements.txt',
  },
  {
    number: '03',
    title: 'Prepare Data',
    description: 'Run the data preparation script to create train/val/test splits',
    code: 'python scripts/prepare_data.py',
  },
  {
    number: '04',
    title: 'Train Your Model',
    description: 'Build and train your GNN (check baseline for reference)',
    code: 'python baselines/simple_gnn.py',
  },
  {
    number: '05',
    title: 'Submit Predictions',
    description: 'Upload your predictions.csv file for evaluation',
    code: 'python scripts/evaluate.py --predictions predictions.csv',
  },
];

const rules = [
  'Maximum 100,000 trainable parameters',
  'Training must complete in under 5 minutes on CPU',
  'No pre-training on external protein/enzyme datasets',
  'No manual feature engineering or domain knowledge injection',
  'Predictions must be in the specified CSV format',
  'One submission per participant per day',
];

export default function DocsPage() {
  return (
    <div className="min-h-screen py-12 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-12">
          <div className="inline-flex items-center gap-2 text-purple-400 mb-4">
            <Book className="w-5 h-5" />
            <span>Documentation</span>
          </div>
          <h1 className="text-4xl font-bold mb-4">Getting Started</h1>
          <p className="text-gray-400 text-lg">
            Everything you need to know to participate in the ENZYMES-Hard challenge.
          </p>
        </div>

        {/* Dataset Overview */}
        <section className="mb-16">
          <h2 className="text-2xl font-bold mb-6 flex items-center gap-3">
            <FileText className="w-6 h-6 text-emerald-400" />
            Dataset Overview
          </h2>
          
          <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6 mb-6">
            <p className="text-gray-300 mb-4">
              The ENZYMES dataset contains <span className="text-emerald-400 font-semibold">600 protein tertiary structures</span> represented 
              as graphs. Each protein belongs to one of <span className="text-purple-400 font-semibold">6 EC top-level enzyme classes</span>.
            </p>
            
            <div className="grid md:grid-cols-2 gap-4 mt-6">
              <div className="p-4 bg-gray-800/50 rounded-lg">
                <h4 className="font-semibold text-gray-200 mb-2">Graph Structure</h4>
                <ul className="text-sm text-gray-400 space-y-1">
                  <li>• Nodes: Secondary structure elements (SSEs)</li>
                  <li>• Edges: Spatial proximity (&lt;6 Ångström)</li>
                  <li>• Total Nodes: 19,580</li>
                  <li>• Total Edges: 74,564</li>
                </ul>
              </div>
              <div className="p-4 bg-gray-800/50 rounded-lg">
                <h4 className="font-semibold text-gray-200 mb-2">Node Features (18-dim)</h4>
                <ul className="text-sm text-gray-400 space-y-1">
                  <li>• Physical properties</li>
                  <li>• Chemical characteristics</li>
                  <li>• Structural attributes</li>
                  <li>• 10-15% missing (NaN) in test</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="grid md:grid-cols-3 gap-4">
            <div className="p-4 bg-gradient-to-br from-purple-900/30 to-purple-900/10 border border-purple-800/30 rounded-xl text-center">
              <div className="text-3xl font-bold text-purple-400 mb-1">240</div>
              <div className="text-sm text-gray-400">Training Graphs</div>
            </div>
            <div className="p-4 bg-gradient-to-br from-emerald-900/30 to-emerald-900/10 border border-emerald-800/30 rounded-xl text-center">
              <div className="text-3xl font-bold text-emerald-400 mb-1">180</div>
              <div className="text-sm text-gray-400">Validation Graphs</div>
              <div className="text-xs text-gray-500 mt-1">(Imbalanced)</div>
            </div>
            <div className="p-4 bg-gradient-to-br from-amber-900/30 to-amber-900/10 border border-amber-800/30 rounded-xl text-center">
              <div className="text-3xl font-bold text-amber-400 mb-1">180</div>
              <div className="text-sm text-gray-400">Test Graphs</div>
              <div className="text-xs text-gray-500 mt-1">(10% edge dropout)</div>
            </div>
          </div>
        </section>

        {/* Quick Start Steps */}
        <section className="mb-16">
          <h2 className="text-2xl font-bold mb-6 flex items-center gap-3">
            <Terminal className="w-6 h-6 text-blue-400" />
            Quick Start
          </h2>
          
          <div className="space-y-6">
            {steps.map((step, idx) => (
              <div
                key={idx}
                className="relative pl-16 pb-6 border-l border-gray-800 last:border-l-0"
              >
                <div className="absolute left-0 -translate-x-1/2 w-8 h-8 bg-purple-600 rounded-full flex items-center justify-center text-sm font-bold">
                  {idx + 1}
                </div>
                <div className="mb-2">
                  <h3 className="text-lg font-semibold text-gray-200">{step.title}</h3>
                  <p className="text-gray-500 text-sm">{step.description}</p>
                </div>
                <div className="code-block p-4 mt-3 overflow-x-auto">
                  <pre className="text-sm text-emerald-400 whitespace-pre">{step.code}</pre>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Submission Format */}
        <section className="mb-16">
          <h2 className="text-2xl font-bold mb-6 flex items-center gap-3">
            <Code className="w-6 h-6 text-amber-400" />
            Submission Format
          </h2>
          
          <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6">
            <p className="text-gray-300 mb-4">
              Your submission must be a CSV file with exactly two columns: <code className="text-purple-400">graph_id</code> and <code className="text-purple-400">prediction</code>.
            </p>
            
            <div className="code-block p-4 overflow-x-auto mb-4">
              <pre className="text-sm">
<span className="text-gray-500"># predictions.csv</span>
<span className="text-purple-400">graph_id</span>,<span className="text-purple-400">prediction</span>
<span className="text-gray-400">0</span>,<span className="text-emerald-400">3</span>
<span className="text-gray-400">1</span>,<span className="text-emerald-400">1</span>
<span className="text-gray-400">2</span>,<span className="text-emerald-400">5</span>
<span className="text-gray-500">...</span>
              </pre>
            </div>
            
            <ul className="text-sm text-gray-400 space-y-2">
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-emerald-400 mt-0.5" />
                <span><code>graph_id</code>: Integer matching the test graph indices (0-179)</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-emerald-400 mt-0.5" />
                <span><code>prediction</code>: Predicted class label (1-6)</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-emerald-400 mt-0.5" />
                <span>Must contain exactly 180 predictions</span>
              </li>
            </ul>
          </div>
        </section>

        {/* Rules */}
        <section className="mb-16">
          <h2 className="text-2xl font-bold mb-6 flex items-center gap-3">
            <CheckCircle className="w-6 h-6 text-red-400" />
            Rules & Constraints
          </h2>
          
          <div className="bg-red-900/10 border border-red-900/30 rounded-xl p-6">
            <ul className="space-y-3">
              {rules.map((rule, idx) => (
                <li key={idx} className="flex items-start gap-3 text-gray-300">
                  <span className="w-6 h-6 flex items-center justify-center bg-red-900/30 rounded text-red-400 text-sm font-bold">
                    {idx + 1}
                  </span>
                  {rule}
                </li>
              ))}
            </ul>
          </div>
        </section>

        {/* Resources */}
        <section>
          <h2 className="text-2xl font-bold mb-6 flex items-center gap-3">
            <Download className="w-6 h-6 text-purple-400" />
            Resources
          </h2>
          
          <div className="grid md:grid-cols-2 gap-4">
            <a
              href="https://github.com/your-org/enzymes-hard-challenge"
              target="_blank"
              rel="noopener noreferrer"
              className="p-4 bg-gray-900/50 border border-gray-800 rounded-xl hover:border-purple-600 transition-colors"
            >
              <h4 className="font-semibold text-gray-200 mb-1">GitHub Repository</h4>
              <p className="text-sm text-gray-500">Starter code, baselines, and scripts</p>
            </a>
            <a
              href="https://pytorch-geometric.readthedocs.io/"
              target="_blank"
              rel="noopener noreferrer"
              className="p-4 bg-gray-900/50 border border-gray-800 rounded-xl hover:border-purple-600 transition-colors"
            >
              <h4 className="font-semibold text-gray-200 mb-1">PyTorch Geometric Docs</h4>
              <p className="text-sm text-gray-500">GNN library documentation</p>
            </a>
          </div>
        </section>

        {/* CTA */}
        <div className="mt-12 text-center">
          <Link
            href="/submit"
            className="inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-purple-600 to-indigo-600 rounded-lg font-semibold 
                     hover:from-purple-500 hover:to-indigo-500 transition-all"
          >
            Submit Your Solution
          </Link>
        </div>
      </div>
    </div>
  );
}
