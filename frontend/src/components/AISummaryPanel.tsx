import { BrainCircuit } from "lucide-react";

interface AISummaryPanelProps {
  summary: string | null;
  visible: boolean;
}

const AISummaryPanel = ({ summary, visible }: AISummaryPanelProps) => {
  if (!visible || !summary) return null;

  return (
    <section className="py-8 pb-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
      <div
        className="glass-card-hover p-6 sm:p-8"
        style={{ animation: "fadeInUp 0.6s ease-out 0.5s both" }}
      >
        <div className="flex items-center gap-2.5 mb-5">
          <div className="w-9 h-9 rounded-xl bg-primary/10 flex items-center justify-center">
            <BrainCircuit className="w-4.5 h-4.5 text-primary" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-foreground">AI Optimization Report</h3>
            <p className="text-xs text-muted-foreground">Intelligent route analysis</p>
          </div>
        </div>

        <div className="rounded-xl bg-secondary/40 border border-border/40 p-5">
          <p className="text-sm leading-relaxed text-secondary-foreground">{summary}</p>
        </div>

        <div className="mt-4 flex items-center gap-4">
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-success" />
            <span className="text-xs text-muted-foreground">Optimization complete</span>
          </div>
          <span className="text-xs text-muted-foreground">•</span>
          <span className="text-xs text-muted-foreground">Confidence: High</span>
        </div>
      </div>
    </section>
  );
};

export default AISummaryPanel;
