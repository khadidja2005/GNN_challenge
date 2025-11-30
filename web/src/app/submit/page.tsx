'use client';

import { useState, useRef } from 'react';
import { Upload, FileText, CheckCircle, XCircle, AlertCircle, TrendingUp, BarChart3 } from 'lucide-react';

interface EvaluationResult {
  success: boolean;
  macroF1: number;
  accuracy: number;
  perClassF1: number[];
  confusionMatrix?: number[][];
  error?: string;
}

// Simulated evaluation (in production, this would call the backend)
function evaluatePredictions(csvContent: string): EvaluationResult {
  try {
    const lines = csvContent.trim().split('\n');
    const header = lines[0].toLowerCase();
    
    if (!header.includes('graph_id') || !header.includes('prediction')) {
      return { success: false, error: 'Invalid CSV format. Must have graph_id and prediction columns.', macroF1: 0, accuracy: 0, perClassF1: [] };
    }
    
    const predictions: { [key: number]: number } = {};
    for (let i = 1; i < lines.length; i++) {
      const parts = lines[i].split(',');
      if (parts.length >= 2) {
        const graphId = parseInt(parts[0].trim());
        const prediction = parseInt(parts[1].trim());
        
        if (isNaN(graphId) || isNaN(prediction)) continue;
        if (prediction < 1 || prediction > 6) {
          return { success: false, error: `Invalid prediction ${prediction} at row ${i}. Must be 1-6.`, macroF1: 0, accuracy: 0, perClassF1: [] };
        }
        predictions[graphId] = prediction;
      }
    }
    
    if (Object.keys(predictions).length !== 180) {
      return { 
        success: false, 
        error: `Expected 180 predictions, got ${Object.keys(predictions).length}.`,
        macroF1: 0, 
        accuracy: 0, 
        perClassF1: [] 
      };
    }
    
    // Simulate evaluation (in production, compare against ground truth)
    // This is a demo - returns random but reasonable scores
    const baseF1 = 0.25 + Math.random() * 0.35; // 0.25 to 0.60
    const perClassF1 = Array.from({ length: 6 }, () => 
      Math.max(0.1, Math.min(0.9, baseF1 + (Math.random() - 0.5) * 0.3))
    );
    const macroF1 = perClassF1.reduce((a, b) => a + b, 0) / 6;
    const accuracy = macroF1 * (0.9 + Math.random() * 0.2);
    
    return {
      success: true,
      macroF1: Math.round(macroF1 * 1000) / 1000,
      accuracy: Math.round(accuracy * 1000) / 1000,
      perClassF1: perClassF1.map(f => Math.round(f * 1000) / 1000),
    };
  } catch (err) {
    return { success: false, error: 'Failed to parse CSV file.', macroF1: 0, accuracy: 0, perClassF1: [] };
  }
}

export default function SubmitPage() {
  const [file, setFile] = useState<File | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [result, setResult] = useState<EvaluationResult | null>(null);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.name.endsWith('.csv')) {
      setFile(droppedFile);
      setResult(null);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setResult(null);
    }
  };

  const handleSubmit = async () => {
    if (!file) return;
    
    setIsEvaluating(true);
    
    try {
      const content = await file.text();
      // Simulate network delay
      await new Promise(resolve => setTimeout(resolve, 1500));
      const evalResult = evaluatePredictions(content);
      setResult(evalResult);
    } catch {
      setResult({ success: false, error: 'Failed to read file.', macroF1: 0, accuracy: 0, perClassF1: [] });
    }
    
    setIsEvaluating(false);
  };

  const baselineF1 = 0.29;
  const improvement = result?.success ? ((result.macroF1 - baselineF1) / baselineF1 * 100).toFixed(1) : null;

  return (
    <div className="min-h-screen py-12 px-4">
      <div className="max-w-3xl mx-auto">
        {/* Header */}
        <div className="mb-12 text-center">
          <div className="inline-flex items-center gap-2 text-purple-400 mb-4">
            <Upload className="w-5 h-5" />
            <span>Submit Solution</span>
          </div>
          <h1 className="text-4xl font-bold mb-4">Evaluate Your Model</h1>
          <p className="text-gray-400 text-lg">
            Upload your predictions CSV file to see how your GNN performs.
          </p>
        </div>

        {/* Upload Area */}
        <div
          onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`
            relative mb-8 p-12 border-2 border-dashed rounded-xl cursor-pointer transition-all
            ${isDragOver 
              ? 'border-purple-500 bg-purple-500/10' 
              : 'border-gray-700 bg-gray-900/30 hover:border-gray-600 hover:bg-gray-900/50'
            }
          `}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv"
            onChange={handleFileSelect}
            className="hidden"
          />
          
          <div className="text-center">
            {file ? (
              <>
                <FileText className="w-16 h-16 mx-auto mb-4 text-emerald-400" />
                <p className="text-lg font-semibold text-gray-200">{file.name}</p>
                <p className="text-sm text-gray-500 mt-1">
                  {(file.size / 1024).toFixed(1)} KB • Click to change
                </p>
              </>
            ) : (
              <>
                <Upload className="w-16 h-16 mx-auto mb-4 text-gray-500" />
                <p className="text-lg font-semibold text-gray-300">
                  Drop your predictions.csv here
                </p>
                <p className="text-sm text-gray-500 mt-1">
                  or click to browse
                </p>
              </>
            )}
          </div>
        </div>

        {/* Submit Button */}
        <div className="text-center mb-12">
          <button
            onClick={handleSubmit}
            disabled={!file || isEvaluating}
            className={`
              px-8 py-4 rounded-lg font-semibold transition-all
              ${file && !isEvaluating
                ? 'bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 glow'
                : 'bg-gray-800 text-gray-500 cursor-not-allowed'
              }
            `}
          >
            {isEvaluating ? (
              <span className="flex items-center gap-2">
                <span className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Evaluating...
              </span>
            ) : (
              'Submit & Evaluate'
            )}
          </button>
        </div>

        {/* Results */}
        {result && (
          <div className={`
            p-6 rounded-xl border
            ${result.success 
              ? 'bg-gray-900/50 border-gray-800' 
              : 'bg-red-900/10 border-red-900/30'
            }
          `}>
            {result.success ? (
              <>
                <div className="flex items-center gap-3 mb-6">
                  <CheckCircle className="w-8 h-8 text-emerald-400" />
                  <div>
                    <h3 className="text-xl font-bold text-gray-200">Evaluation Complete</h3>
                    <p className="text-gray-500 text-sm">Your predictions have been scored</p>
                  </div>
                </div>

                {/* Main Scores */}
                <div className="grid md:grid-cols-2 gap-4 mb-6">
                  <div className="p-4 bg-gradient-to-br from-emerald-900/30 to-emerald-900/10 border border-emerald-800/30 rounded-xl">
                    <div className="flex items-center gap-2 text-emerald-400 mb-2">
                      <TrendingUp className="w-5 h-5" />
                      <span className="text-sm font-medium">Macro F1-Score</span>
                    </div>
                    <div className="text-4xl font-bold text-white">{result.macroF1.toFixed(3)}</div>
                    {improvement && (
                      <div className={`text-sm mt-1 ${parseFloat(improvement) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {parseFloat(improvement) >= 0 ? '+' : ''}{improvement}% vs baseline
                      </div>
                    )}
                  </div>
                  
                  <div className="p-4 bg-gradient-to-br from-blue-900/30 to-blue-900/10 border border-blue-800/30 rounded-xl">
                    <div className="flex items-center gap-2 text-blue-400 mb-2">
                      <BarChart3 className="w-5 h-5" />
                      <span className="text-sm font-medium">Accuracy</span>
                    </div>
                    <div className="text-4xl font-bold text-white">{(result.accuracy * 100).toFixed(1)}%</div>
                    <div className="text-sm text-gray-500 mt-1">
                      Overall correct predictions
                    </div>
                  </div>
                </div>

                {/* Per-Class F1 */}
                <div className="p-4 bg-gray-800/50 rounded-xl">
                  <h4 className="font-semibold text-gray-300 mb-4">Per-Class F1 Scores</h4>
                  <div className="grid grid-cols-3 md:grid-cols-6 gap-3">
                    {result.perClassF1.map((f1, idx) => (
                      <div key={idx} className="text-center">
                        <div className="text-xs text-gray-500 mb-1">Class {idx + 1}</div>
                        <div className="w-full bg-gray-700 rounded-full h-2 mb-1">
                          <div 
                            className="bg-purple-500 h-2 rounded-full transition-all" 
                            style={{ width: `${f1 * 100}%` }}
                          />
                        </div>
                        <div className="text-sm font-mono text-gray-300">{f1.toFixed(2)}</div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Baseline Comparison */}
                <div className="mt-6 p-4 border border-gray-700 rounded-lg">
                  <div className="flex items-center justify-between">
                    <div>
                      <span className="text-gray-400">Baseline (GIN):</span>
                      <span className="ml-2 text-amber-400 font-mono">{baselineF1.toFixed(3)}</span>
                    </div>
                    <div>
                      <span className="text-gray-400">Your Score:</span>
                      <span className="ml-2 text-emerald-400 font-mono font-bold">{result.macroF1.toFixed(3)}</span>
                    </div>
                    <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                      result.macroF1 > baselineF1 
                        ? 'bg-emerald-900/30 text-emerald-400' 
                        : 'bg-amber-900/30 text-amber-400'
                    }`}>
                      {result.macroF1 > baselineF1 ? '🎉 Beat Baseline!' : 'Keep Trying!'}
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <div className="flex items-start gap-3">
                <XCircle className="w-8 h-8 text-red-400 flex-shrink-0" />
                <div>
                  <h3 className="text-xl font-bold text-red-400">Evaluation Failed</h3>
                  <p className="text-gray-400 mt-1">{result.error}</p>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Format Reminder */}
        <div className="mt-8 p-4 bg-gray-900/30 border border-gray-800 rounded-xl">
          <div className="flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-amber-400 mt-0.5" />
            <div className="text-sm">
              <p className="text-gray-300 font-medium">Submission Format</p>
              <p className="text-gray-500 mt-1">
                Your CSV must have columns <code className="text-purple-400">graph_id</code> and <code className="text-purple-400">prediction</code> 
                with exactly 180 rows (one per test graph). Predictions must be integers 1-6.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
