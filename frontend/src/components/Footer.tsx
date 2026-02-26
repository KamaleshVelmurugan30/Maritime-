import { Anchor } from "lucide-react";

const Footer = () => (
  <footer className="border-t border-border/40 py-6 px-4 sm:px-6 lg:px-8">
    <div className="max-w-7xl mx-auto flex items-center justify-between">
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <Anchor className="w-3.5 h-3.5 text-primary" />
        <span>NaviGreen AI</span>
      </div>
      <span className="text-xs text-muted-foreground">© 2026 NaviGreen AI. All rights reserved.</span>
    </div>
  </footer>
);

export default Footer;
