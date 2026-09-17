'use client';

import { useMemo, useState } from 'react';
import { ArrowRight, Copy, FileText, LoaderCircle, RotateCcw } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';

type Method = 'extractive' | 'abstractive';

type SummaryResponse = {
  summary: string;
  stats: {
    original_words: number;
    summary_words: number;
  };
};

const SAMPLE = `Artificial intelligence is increasingly used to support medical diagnosis, but researchers warn that successful lab results do not always translate to hospitals. A model trained at one institution may perform differently when patient demographics, equipment, or clinical workflows change. To address this, several hospitals are testing systems prospectively before using their recommendations in patient care. These evaluations measure not only accuracy, but also whether clinicians understand and appropriately act on a prediction. Researchers argue that transparent reporting and continued monitoring are essential because medical data changes over time. They also emphasize that AI tools should assist trained professionals rather than replace clinical judgment.`;
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

export default function Home() {
  const [text, setText] = useState('');
  const [method, setMethod] = useState<Method>('extractive');
  const [length, setLength] = useState(80);
  const [result, setResult] = useState<SummaryResponse | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const inputWords = useMemo(() => text.trim().split(/\s+/).filter(Boolean).length, [text]);

  async function summarize() {
    if (!text.trim() || loading) return;
    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/summarize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, method, length }),
      });
      const body = await response.json() as SummaryResponse & { detail?: unknown };
      if (!response.ok) {
        throw new Error(typeof body.detail === 'string' ? body.detail : 'The summary could not be generated.');
      }
      setResult(body);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not reach the local API.');
    } finally {
      setLoading(false);
    }
  }

  async function copySummary() {
    if (!result) return;

    try {
      await navigator.clipboard.writeText(result.summary);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    } catch {
      setError('Clipboard access was blocked. Select and copy the summary manually.');
    }
  }

  function reset() {
    setText('');
    setResult(null);
    setError('');
  }

  return (
    <main className="min-h-screen bg-background text-foreground selection:bg-primary/25 selection:text-foreground">
      <div className="mx-auto flex min-h-screen w-full max-w-[1480px] flex-col px-4 py-5 sm:px-7 sm:py-7 lg:px-10 lg:py-8">
        <header className="flex items-center border-b border-border pb-6">
          <div className="flex items-center gap-3">
            <span className="grid size-11 place-items-center rounded-full bg-primary text-primary-foreground">
              <FileText aria-hidden="true" className="size-5" strokeWidth={1.8} />
            </span>
            <div>
              <p className="font-heading text-2xl font-semibold tracking-[-0.02em]">SummarizeX2</p>
              <p className="mt-0.5 text-sm text-muted-foreground">Turn long text into a quick summary.</p>
            </div>
          </div>
        </header>

        <section className="grid flex-1 gap-5 py-5 lg:grid-cols-2">
          <article className="paper-panel flex min-h-[590px] flex-col">
            <div className="flex items-start justify-between border-b border-border px-5 py-5 sm:px-6">
              <div>
                <p className="section-label">Original</p>
                <h1 className="mt-1.5 font-heading text-2xl font-semibold tracking-[-0.025em]">Paste your text</h1>
              </div>
              <Button aria-label="Clear text" className="rounded-full text-muted-foreground hover:bg-secondary hover:text-foreground" disabled={!text} onClick={reset} size="icon" variant="ghost">
                <RotateCcw aria-hidden="true" />
              </Button>
            </div>

            <textarea
              aria-label="Source text"
              className="min-h-[310px] flex-1 resize-none bg-transparent px-5 py-6 text-[16px] leading-7 text-foreground outline-none placeholder:text-muted-foreground/60 sm:px-6"
              maxLength={50000}
              onChange={(event) => setText(event.target.value)}
              placeholder="Paste an article, report, or passage…"
              spellCheck="true"
              value={text}
            />

            <div className="border-t border-border p-5 sm:p-6">
              <div className="mb-5 flex items-center justify-between gap-4">
                <button className="text-sm font-medium text-primary underline decoration-primary/30 underline-offset-4 transition-colors hover:decoration-primary" onClick={() => setText(SAMPLE)} type="button">
                  Try an example
                </button>
                <span className="font-mono text-xs tabular-nums text-muted-foreground">{inputWords.toLocaleString()} words</span>
              </div>

              <div className="grid gap-6 border-t border-border/70 pt-5 sm:grid-cols-[1.25fr_1fr]">
                <div>
                  <p className="control-label">Summary style</p>
                  <Tabs onValueChange={(value) => setMethod(value as Method)} value={method}>
                    <TabsList className="mt-2 h-auto w-full gap-1 rounded-xl bg-secondary p-1.5">
                      <TabsTrigger className="h-auto min-w-0 flex-col items-start gap-0.5 rounded-lg px-3 py-2.5 text-left data-active:bg-card data-active:shadow-sm" value="extractive">
                        <span className="text-sm font-semibold">Extract</span>
                        <span className="text-[11px] font-normal text-muted-foreground">Pulls key sentences</span>
                      </TabsTrigger>
                      <TabsTrigger className="h-auto min-w-0 flex-col items-start gap-0.5 rounded-lg px-3 py-2.5 text-left data-active:bg-card data-active:shadow-sm" value="abstractive">
                        <span className="text-sm font-semibold">Rewrite</span>
                        <span className="text-[11px] font-normal text-muted-foreground">Writes it fresh</span>
                      </TabsTrigger>
                    </TabsList>
                  </Tabs>
                </div>

                <div>
                  <div className="flex items-center justify-between gap-3">
                    <label className="control-label" htmlFor="summary-length">Target length</label>
                    <span className="font-mono text-xs tabular-nums text-primary">{length} words</span>
                  </div>
                  <Slider aria-label="Target summary length" className="mt-6" id="summary-length" max={250} min={20} onValueChange={(value) => setLength(Array.isArray(value) ? value[0] : value)} step={5} value={[length]} />
                </div>
              </div>

              <Button className="mt-6 h-12 w-full rounded-xl bg-primary text-[15px] font-semibold text-primary-foreground shadow-[0_8px_20px_rgba(133,68,42,0.16)] hover:bg-primary/90" disabled={!text.trim() || loading} onClick={summarize}>
                {loading ? <LoaderCircle className="animate-spin" /> : <ArrowRight />}
                {loading ? 'Shortening…' : 'Shorten text'}
              </Button>
            </div>
          </article>

          <article className="paper-panel flex min-h-[590px] flex-col">
            <div className="flex items-start justify-between border-b border-border px-5 py-5 sm:px-6">
              <div>
                <p className="section-label">Result</p>
                <h2 className="mt-1.5 font-heading text-2xl font-semibold tracking-[-0.025em]">Your summary</h2>
              </div>
              <Button aria-label="Copy summary" className="h-9 rounded-full border-border bg-transparent px-3.5 text-xs text-muted-foreground hover:bg-secondary hover:text-foreground" disabled={!result} onClick={copySummary} variant="outline">
                <Copy aria-hidden="true" />
                {copied ? 'Copied' : 'Copy'}
              </Button>
            </div>

            <div className="summary-scroll relative flex min-h-[370px] flex-1 flex-col overflow-y-auto p-5 sm:p-7">
              {error ? (
                <div className="rounded-r-lg border-l-2 border-destructive bg-destructive/8 px-4 py-3 text-sm leading-6 text-destructive" role="alert">{error}</div>
              ) : result ? (
                <p className="max-w-[68ch] font-heading text-[18px] leading-8 text-foreground/90">{result.summary}</p>
              ) : (
                <div className="m-auto max-w-sm text-center">
                  <FileText aria-hidden="true" className="mx-auto mb-4 size-8 text-muted-foreground/35" strokeWidth={1.4} />
                  <p className="text-sm leading-6 text-muted-foreground">Your summary will show up here.</p>
                </div>
              )}
            </div>

            <div className="grid grid-cols-2 border-t border-border">
              <Stat label="Original" value={result ? `${result.stats.original_words}` : '—'} suffix="words" />
              <Stat label="Summary" value={result ? `${result.stats.summary_words}` : '—'} suffix="words" />
            </div>
          </article>
        </section>
      </div>
    </main>
  );
}

function Stat({ label, value, suffix }: { label: string; value: string; suffix: string }) {
  return (
    <div className="min-h-24 border-r border-border p-4 even:border-r-0 sm:even:border-r sm:last:border-r-0">
      <p className="text-[11px] font-medium uppercase tracking-[0.12em] text-muted-foreground">{label}</p>
      <p className="mt-2 font-mono text-xl font-medium tabular-nums">
        {value} {suffix && <span className="font-sans text-xs font-normal text-muted-foreground">{suffix}</span>}
      </p>
    </div>
  );
}
