import { Anchor } from "lucide-react";
import { Link, useLocation } from "react-router-dom";

const navLinks = [
  { to: "/dashboard", label: "Dashboard" },
  { to: "/analytics", label: "Analytics" },
  { to: "/fleet", label: "Fleet" },
];

const Navbar = () => {
  const { pathname } = useLocation();

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 backdrop-blur-xl border-b border-border/50"
      style={{ background: "hsl(220 25% 6% / 0.85)" }}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <Link to="/" className="flex items-center gap-2.5">
            <Anchor className="w-5 h-5 text-primary" />
            <span className="text-lg font-bold tracking-tight text-foreground">
              NaviGreen
            </span>
            <span className="text-lg">🌊</span>
          </Link>

          <div className="hidden md:flex items-center gap-8">
            {navLinks.map((link) => (
              <Link
                key={link.to}
                to={link.to}
                className={`text-sm transition-colors ${
                  pathname === link.to ? "text-foreground font-medium" : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {link.label}
              </Link>
            ))}
          </div>

          <div className="w-[120px]" />
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
