import { useEffect, useMemo, useState } from "react";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import { Ship, Search, Plus, Fuel, Clock, Anchor, Waves } from "lucide-react";

type VesselStatus = "En Route" | "Optimizing" | "Docked";

type Port = {
  lat: number;
  lon: number;
};

type Vessel = {
  id: string;
  name: string;
  status: VesselStatus;
  fuel: number;
  eta_hours: number | null;
  start_port?: Port;
  end_port?: Port;
};

const API_BASE = "http://127.0.0.1:8000";

const statusStyles: Record<
  VesselStatus,
  { badge: string; dot: string; label: string }
> = {
  "En Route": {
    badge: "bg-emerald-500/15 text-emerald-300 ring-1 ring-emerald-500/30",
    dot: "bg-emerald-400",
    label: "En Route",
  },
  Optimizing: {
    badge: "bg-amber-500/15 text-amber-300 ring-1 ring-amber-500/30",
    dot: "bg-amber-400",
    label: "Optimizing",
  },
  Docked: {
    badge: "bg-slate-500/15 text-slate-200 ring-1 ring-slate-500/30",
    dot: "bg-slate-300",
    label: "Docked",
  },
};

function clamp(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n));
}

function fuelBarClass(fuel: number) {
  if (fuel >= 60) return "bg-emerald-400";
  if (fuel >= 30) return "bg-amber-400";
  return "bg-rose-400";
}

const Fleet = () => {
  const [vessels, setVessels] = useState<Vessel[]>([]);
  const [loading, setLoading] = useState(false);

  // form state
  const [name, setName] = useState("");
  const [status, setStatus] = useState<VesselStatus>("En Route");
  const [fuel, setFuel] = useState("");
  const [eta, setEta] = useState("");
  const [startLat, setStartLat] = useState("");
  const [startLon, setStartLon] = useState("");
  const [endLat, setEndLat] = useState("");
  const [endLon, setEndLon] = useState("");
  // UI controls
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<"All" | VesselStatus>("All");
  const [sortBy, setSortBy] = useState<"Name" | "Fuel" | "ETA">("Name");

  const loadFleet = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/fleet`);
      const data = await res.json();
      setVessels(Array.isArray(data) ? data : []);
    } catch (e) {
      console.error("Failed to load fleet:", e);
      setVessels([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadFleet();
  }, []);

  const addVessel = async (e: React.FormEvent) => {
    e.preventDefault();

    const newVessel: Vessel = {
      id: Date.now().toString(),
      name: name.trim(),
      status,
      fuel: clamp(Number(fuel), 0, 100),
      eta_hours: eta ? Math.max(0, Number(eta)) : null,
    };

    try {
      await fetch(`${API_BASE}/fleet`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(newVessel),
      });

      setName("");
      setFuel("");
      setEta("");
      setStatus("En Route");
      loadFleet();
    } catch (err) {
      console.error("Failed to add vessel:", err);
    }
  };

  // ---- derived stats ----
  const stats = useMemo(() => {
    const total = vessels.length;
    const enRoute = vessels.filter((v) => v.status === "En Route").length;
    const optimizing = vessels.filter((v) => v.status === "Optimizing").length;
    const docked = vessels.filter((v) => v.status === "Docked").length;
    const avgFuel =
      total === 0
        ? 0
        : Math.round(
            vessels.reduce((acc, v) => acc + (Number(v.fuel) || 0), 0) / total
          );
    const withEta = vessels.filter((v) => v.eta_hours !== null);
    const avgEta =
      withEta.length === 0
        ? null
        : Number(
            (
              withEta.reduce((acc, v) => acc + (v.eta_hours || 0), 0) /
              withEta.length
            ).toFixed(1)
          );
    return { total, enRoute, optimizing, docked, avgFuel, avgEta };
  }, [vessels]);

  const visible = useMemo(() => {
    let list = [...vessels];

    if (filter !== "All") {
      list = list.filter((v) => v.status === filter);
    }

    const q = query.trim().toLowerCase();
    if (q) {
      list = list.filter((v) => v.name.toLowerCase().includes(q));
    }

    list.sort((a, b) => {
      if (sortBy === "Name") return a.name.localeCompare(b.name);
      if (sortBy === "Fuel") return (b.fuel ?? 0) - (a.fuel ?? 0);
      // ETA: nulls go last
      const aEta = a.eta_hours ?? Number.POSITIVE_INFINITY;
      const bEta = b.eta_hours ?? Number.POSITIVE_INFINITY;
      return aEta - bEta;
    });

    return list;
  }, [vessels, filter, query, sortBy]);

  return (
    <div className="min-h-screen ocean-gradient flex flex-col">
      <Navbar />

      <main className="flex-1 pt-20 px-6 max-w-7xl mx-auto w-full">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-3 mb-6">
          <div>
            <div className="inline-flex items-center gap-2 text-xs px-3 py-1 rounded-full bg-white/5 ring-1 ring-white/10 text-muted-foreground">
              <Waves className="w-4 h-4" />
              Fleet Control Center
            </div>
            <h1 className="text-2xl md:text-3xl font-bold mt-2">
              Live Fleet Overview
            </h1>
            <p className="text-sm text-muted-foreground mt-1">
              Add vessels, track fuel & ETA, and monitor statuses in one view.
            </p>
          </div>

          <button
            onClick={loadFleet}
            className="btn-primary inline-flex items-center gap-2 self-start md:self-auto"
            type="button"
          >
            Refresh
          </button>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4 mb-8">
          <div className="glass-card-hover p-5">
            <div className="flex items-center justify-between">
              <div className="text-sm text-muted-foreground">Total Vessels</div>
              <Ship className="w-5 h-5 text-primary" />
            </div>
            <div className="text-2xl font-bold mt-2">{stats.total}</div>
          </div>

          <div className="glass-card-hover p-5">
            <div className="flex items-center justify-between">
              <div className="text-sm text-muted-foreground">En Route</div>
              <div className="w-2.5 h-2.5 rounded-full bg-emerald-400" />
            </div>
            <div className="text-2xl font-bold mt-2">{stats.enRoute}</div>
          </div>

          <div className="glass-card-hover p-5">
            <div className="flex items-center justify-between">
              <div className="text-sm text-muted-foreground">Optimizing</div>
              <div className="w-2.5 h-2.5 rounded-full bg-amber-400" />
            </div>
            <div className="text-2xl font-bold mt-2">{stats.optimizing}</div>
          </div>

          <div className="glass-card-hover p-5">
            <div className="flex items-center justify-between">
              <div className="text-sm text-muted-foreground">Docked</div>
              <Anchor className="w-5 h-5 text-slate-200/70" />
            </div>
            <div className="text-2xl font-bold mt-2">{stats.docked}</div>
          </div>

          <div className="glass-card-hover p-5">
            <div className="flex items-center justify-between">
              <div className="text-sm text-muted-foreground">Avg Fuel</div>
              <Fuel className="w-5 h-5 text-primary" />
            </div>
            <div className="text-2xl font-bold mt-2">{stats.avgFuel}%</div>
            <div className="text-xs text-muted-foreground mt-1">
              Avg ETA: {stats.avgEta !== null ? `${stats.avgEta}h` : "—"}
            </div>
          </div>
        </div>

        {/* Layout: Form + List */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Add Vessel */}
          <section className="lg:col-span-4">
            <div className="glass-card-hover p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-bold">Add Vessel</h2>
                <div className="inline-flex items-center gap-2 text-xs text-muted-foreground">
                  <Plus className="w-4 h-4" /> New entry
                </div>
              </div>

              <form onSubmit={addVessel} className="grid gap-4">
                <div className="grid gap-2">
                  <label className="text-xs text-muted-foreground">
                    Vessel Name
                  </label>
                  <input
                    className="input"
                    placeholder="Eg: Aurora-07"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    required
                  />
                </div>

                <div className="grid gap-2">
                  <label className="text-xs text-muted-foreground">Status</label>
                  <select
                    className="input"
                    value={status}
                    onChange={(e) => setStatus(e.target.value as VesselStatus)}
                  >
                    <option value="En Route">En Route</option>
                    <option value="Optimizing">Optimizing</option>
                    <option value="Docked">Docked</option>
                  </select>
                </div>

                <div className="grid gap-2">
                  <label className="text-xs text-muted-foreground">
                    Fuel (%)
                  </label>
                  <input
                    className="input"
                    placeholder="0 - 100"
                    type="number"
                    min={0}
                    max={100}
                    value={fuel}
                    onChange={(e) => setFuel(e.target.value)}
                    required
                  />
                </div>

                <div className="grid gap-2">
                  <label className="text-xs text-muted-foreground">
                    ETA (hours) - optional
                  </label>
                  <input
                    className="input"
                    placeholder="Eg: 12"
                    type="number"
                    min={0}
                    value={eta}
                    onChange={(e) => setEta(e.target.value)}
                  />
                </div>

                <button
                  type="submit"
                  className="btn-primary inline-flex items-center justify-center gap-2"
                >
                  <Plus className="w-4 h-4" /> Add Vessel
                </button>

                <p className="text-xs text-muted-foreground">
                  Tip: Use “Optimizing” while running route optimization.
                </p>
              </form>
            </div>
          </section>

          {/* Fleet List */}
          <section className="lg:col-span-8">
            <div className="glass-card-hover p-6">
              <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-5">
                <div>
                  <h2 className="text-lg font-bold">Fleet</h2>
                  <p className="text-sm text-muted-foreground">
                    Showing {visible.length} vessel(s)
                    {loading ? " • Loading..." : ""}
                  </p>
                </div>

                <div className="flex flex-col sm:flex-row gap-3 w-full md:w-auto">
                  {/* Search */}
                  <div className="relative w-full sm:w-64">
                    <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
                    <input
                      className="input pl-9"
                      placeholder="Search vessel..."
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                    />
                  </div>

                  {/* Filter */}
                  <select
                    className="input w-full sm:w-44"
                    value={filter}
                    onChange={(e) =>
                      setFilter(e.target.value as "All" | VesselStatus)
                    }
                  >
                    <option value="All">All Status</option>
                    <option value="En Route">En Route</option>
                    <option value="Optimizing">Optimizing</option>
                    <option value="Docked">Docked</option>
                  </select>

                  {/* Sort */}
                  <select
                    className="input w-full sm:w-40"
                    value={sortBy}
                    onChange={(e) =>
                      setSortBy(e.target.value as "Name" | "Fuel" | "ETA")
                    }
                  >
                    <option value="Name">Sort: Name</option>
                    <option value="Fuel">Sort: Fuel</option>
                    <option value="ETA">Sort: ETA</option>
                  </select>
                </div>
              </div>

              {/* List */}
              <div className="grid gap-4">
                {visible.length === 0 ? (
                  <div className="p-10 text-center rounded-2xl bg-white/5 ring-1 ring-white/10">
                    <Ship className="w-8 h-8 mx-auto text-primary" />
                    <div className="font-semibold mt-3">No vessels found</div>
                    <div className="text-sm text-muted-foreground mt-1">
                      Add a vessel or clear filters/search.
                    </div>
                  </div>
                ) : (
                  visible.map((v) => {
                    const st = statusStyles[v.status];
                    const fuelPct = clamp(Number(v.fuel) || 0, 0, 100);

                    return (
                      <div
                        key={v.id}
                        className="rounded-2xl p-5 bg-white/5 ring-1 ring-white/10 hover:bg-white/7 transition"
                      >
                        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                          <div className="flex items-start gap-4">
                            <div className="w-11 h-11 rounded-2xl bg-white/5 ring-1 ring-white/10 flex items-center justify-center">
                              <Ship className="w-5 h-5 text-primary" />
                            </div>

                            <div>
                              <div className="flex items-center gap-3">
                                <div className="text-base font-semibold">
                                  {v.name}
                                </div>

                                <span
                                  className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs ${st.badge}`}
                                >
                                  <span
                                    className={`w-2 h-2 rounded-full ${st.dot}`}
                                  />
                                  {st.label}
                                </span>
                              </div>

                              <div className="text-xs text-muted-foreground mt-1">
                                Vessel ID:{" "}
                                <span className="text-slate-200/90">{v.id}</span>
                              </div>
                            </div>
                          </div>

                          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 sm:gap-6 text-sm">
                            <div className="flex items-center gap-2">
                              <Fuel className="w-4 h-4 text-muted-foreground" />
                              <div>
                                <div className="text-muted-foreground text-xs">
                                  Fuel
                                </div>
                                <div className="font-semibold">{fuelPct}%</div>
                              </div>
                            </div>

                            <div className="flex items-center gap-2">
                              <Clock className="w-4 h-4 text-muted-foreground" />
                              <div>
                                <div className="text-muted-foreground text-xs">
                                  ETA
                                </div>
                                <div className="font-semibold">
                                  {v.eta_hours !== null ? `${v.eta_hours}h` : "—"}
                                </div>
                              </div>
                            </div>

                            <div className="hidden sm:flex items-center gap-2">
                              <Anchor className="w-4 h-4 text-muted-foreground" />
                              <div>
                                <div className="text-muted-foreground text-xs">
                                  State
                                </div>
                                <div className="font-semibold">{v.status}</div>
                              </div>
                            </div>
                          </div>
                        </div>

                        {/* Fuel bar */}
                        <div className="mt-4">
                          <div className="flex items-center justify-between text-xs text-muted-foreground mb-2">
                            <span>Fuel level</span>
                            <span>{fuelPct}%</span>
                          </div>
                          <div className="h-2 rounded-full bg-white/5 ring-1 ring-white/10 overflow-hidden">
                            <div
                              className={`h-full ${fuelBarClass(
                                fuelPct
                              )} rounded-full`}
                              style={{ width: `${fuelPct}%` }}
                            />
                          </div>
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          </section>
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default Fleet;