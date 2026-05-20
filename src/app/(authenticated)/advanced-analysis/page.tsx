"use client";

import { useMemo, useState } from "react";
import {
  AlertCircle,
  BrainCircuit,
  CheckCircle2,
  FileJson,
  FolderOpen,
  Loader2,
  Upload,
  XCircle,
} from "lucide-react";

type ModuleResult = {
  status: "completed" | "failed";
  data?: any;
  error?: string;
};

type AdvancedResult = {
  generated_at: string;
  dataset: string;
  output_dir: string;
  document_count: number;
  source_distribution: Record<string, number>;
  modules: Record<string, ModuleResult>;
};

const moduleLabels: Record<string, string> = {
  linguistic: "Indicateurs linguistiques",
  emotional: "Intensite emotionnelle",
  topics: "Topics BERTopic",
};

export default function AdvancedAnalysisPage() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<AdvancedResult | null>(null);
  const [logs, setLogs] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const modules = useMemo(() => {
    if (!result) return [];
    return Object.entries(result.modules || {});
  }, [result]);

  const handleSubmit = async () => {
    if (!file) {
      setError("Selectionnez un fichier JSON.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);
    setLogs("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("/api/analysis/upload", {
        method: "POST",
        body: formData,
      });
      const payload = await response.json();

      if (!response.ok) {
        throw new Error(payload.error || "Analyse impossible.");
      }

      setResult(payload.result);
      setLogs(payload.logs || "");
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-4xl font-extrabold tracking-tight text-white">Analyse avancee</h1>
          <p className="text-slate-400">
            Importez un JSON et lancez les analyses linguistiques, emotionnelles et topic modeling.
          </p>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
        <section className="rounded-lg border border-slate-900 bg-slate-900/40 p-6">
          <div className="mb-5 flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-violet-500/10 text-violet-300">
              <FileJson className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white">Import JSON</h2>
              <p className="text-sm text-slate-400">Format liste, dataset exporte, ou objet avec articles/tweets.</p>
            </div>
          </div>

          <label className="flex min-h-36 cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed border-slate-700 bg-slate-950/50 px-4 py-8 text-center transition hover:border-violet-500/70">
            <Upload className="mb-3 h-8 w-8 text-slate-400" />
            <span className="text-sm font-semibold text-white">
              {file ? file.name : "Choisir un fichier .json"}
            </span>
            <span className="mt-1 text-xs text-slate-500">
              {file ? `${Math.ceil(file.size / 1024)} KB` : "Le fichier sera analyse cote serveur"}
            </span>
            <input
              type="file"
              accept="application/json,.json"
              className="hidden"
              onChange={(event) => setFile(event.target.files?.[0] || null)}
            />
          </label>

          {error && (
            <div className="mt-4 flex items-start gap-2 rounded-lg border border-red-500/20 bg-red-500/10 p-3 text-sm text-red-300">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <button
            onClick={handleSubmit}
            disabled={loading}
            className="mt-5 flex w-full items-center justify-center gap-2 rounded-lg bg-violet-600 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-violet-500 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <BrainCircuit className="h-4 w-4" />}
            Lancer l'analyse
          </button>
        </section>

        <section className="rounded-lg border border-slate-900 bg-slate-900/30 p-6">
          <div className="mb-5 flex items-center justify-between gap-3">
            <div>
              <h2 className="text-lg font-bold text-white">Resultats</h2>
              <p className="text-sm text-slate-400">
                Les sorties detaillees sont stockees dans le dossier Python genere.
              </p>
            </div>
            {result && (
              <span className="rounded-full border border-slate-700 px-3 py-1 text-xs font-semibold text-slate-300">
                {result.document_count} docs
              </span>
            )}
          </div>

          {!result && !loading && (
            <div className="flex h-64 flex-col items-center justify-center rounded-lg border border-dashed border-slate-800 text-center">
              <FolderOpen className="mb-3 h-9 w-9 text-slate-600" />
              <p className="text-sm text-slate-500">Aucun resultat pour le moment.</p>
            </div>
          )}

          {loading && (
            <div className="flex h-64 flex-col items-center justify-center text-center">
              <Loader2 className="mb-3 h-8 w-8 animate-spin text-violet-400" />
              <p className="text-sm text-slate-400">Analyse en cours...</p>
            </div>
          )}

          {result && (
            <div className="space-y-5">
              <div className="grid gap-3 sm:grid-cols-2">
                {Object.entries(result.source_distribution || {}).map(([source, count]) => (
                  <div key={source} className="rounded-lg border border-slate-800 bg-slate-950/50 p-4">
                    <span className="text-xs uppercase tracking-wider text-slate-500">{source}</span>
                    <div className="mt-1 text-2xl font-bold text-white">{count}</div>
                  </div>
                ))}
              </div>

              <div className="space-y-3">
                {modules.map(([name, module]) => (
                  <div key={name} className="rounded-lg border border-slate-800 bg-slate-950/50 p-4">
                    <div className="flex items-center justify-between gap-3">
                      <div>
                        <h3 className="font-semibold text-white">{moduleLabels[name] || name}</h3>
                        <p className="text-xs text-slate-500">{name}</p>
                      </div>
                      {module.status === "completed" ? (
                        <CheckCircle2 className="h-5 w-5 text-emerald-400" />
                      ) : (
                        <XCircle className="h-5 w-5 text-red-400" />
                      )}
                    </div>
                    {module.error && (
                      <p className="mt-3 rounded-md bg-red-500/10 p-2 text-xs text-red-300">{module.error}</p>
                    )}
                  </div>
                ))}
              </div>

              <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-4">
                <span className="text-xs uppercase tracking-wider text-slate-500">Dossier outputs</span>
                <p className="mt-1 break-all text-sm text-slate-300">{result.output_dir}</p>
              </div>
            </div>
          )}
        </section>
      </div>

      {logs && (
        <section className="rounded-lg border border-slate-900 bg-slate-950 p-4">
          <h2 className="mb-3 text-sm font-bold uppercase tracking-wider text-slate-400">Logs Python</h2>
          <pre className="max-h-72 overflow-auto whitespace-pre-wrap text-xs leading-relaxed text-slate-300">
            {logs}
          </pre>
        </section>
      )}
    </div>
  );
}
