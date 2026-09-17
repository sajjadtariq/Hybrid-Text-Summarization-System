import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'SummarizeX2 — Text Summarizer',
  description: 'Turn long text into a quick summary.',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
