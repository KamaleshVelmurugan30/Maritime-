// import Navbar from "@/components/Navbar";
// import Footer from "@/components/Footer";
// import { BarChart3, TrendingUp, Activity, Globe } from "lucide-react";
// import { useOptimization } from "@/context/OptimizationContext";
// const { data } = useOptimization();
// const stats = data
//   ? [
//       { icon: BarChart3, label: "Fuel Saved", value: `${data.fuelSaved}%`, change: "" },
//       { icon: TrendingUp, label: "CO₂ Reduction", value: `${data.co2Reduction} t`, change: "" },
//       { icon: Activity, label: "Time Saved", value: `${data.timeSaved} hrs`, change: "" },
//       { icon: Globe, label: "Distance Rerouted", value: `${data.distanceRerouted} nm`, change: "" },
//     ]
//   : [];

// const Analytics = () => {
  
//   return (
//     <div className="min-h-screen ocean-gradient flex flex-col">
//       <Navbar />
//       <main className="flex-1 pt-20">
//         <section className="px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto pt-8 pb-4">
//           <h1 className="text-2xl sm:text-3xl font-bold text-foreground mb-1">Analytics</h1>
//           <p className="text-sm text-muted-foreground">Historical performance metrics</p>
//         </section>

//         <section className="py-8 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
//           <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
//             {stats.map((s, i) => (
//               <div
//                 key={s.label}
//                 className="glass-card-hover p-6"
//                 style={{ animation: `slideUp 0.5s ease-out ${i * 100}ms both` }}
//               >
//                 <div className="flex items-center gap-3 mb-4">
//                   <div className="w-9 h-9 rounded-xl bg-primary/10 flex items-center justify-center">
//                     <s.icon className="w-4 h-4 text-primary" />
//                   </div>
//                   <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">{s.label}</span>
//                 </div>
//                 <div className="flex items-baseline gap-2">
//                   <span className="text-3xl font-bold text-foreground">{s.value}</span>
//                   <span className="text-xs text-success font-medium">{s.change}</span>
//                 </div>
//               </div>
//             ))}
//           </div>
//         </section>

//         <section className="py-8 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
//           <div className="glass-card p-6 sm:p-8 min-h-[300px] flex items-center justify-center">
//             <p className="text-sm text-muted-foreground">Charts and trend visualization — ready for Recharts integration</p>
//           </div>
//         </section>
//       </main>
//       <Footer />
//     </div>
//   );
// };

// export default Analytics;
// import Navbar from "@/components/Navbar";




import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import { useOptimization } from "@/context/OptimizationContext";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
} from "recharts";

const Analytics = () => {
  const { data } = useOptimization();

  if (!data) {
    return (
      <div className="min-h-screen ocean-gradient flex flex-col">
        <Navbar />
        <main className="flex-1 flex items-center justify-center">
          <p className="text-muted-foreground">
            Run optimization to see analytics
          </p>
        </main>
        <Footer />
      </div>
    );
  }

  const chartData = Object.entries(data.routeComparison || {}).map(
    ([route, values]: any) => ({
      route,
      distance: values.distance,
      fuel: values.fuel,
    })
  );

  return (
    <div className="min-h-screen ocean-gradient flex flex-col">
      <Navbar />
      <main className="flex-1 pt-20 px-6 max-w-7xl mx-auto">

        <h1 className="text-2xl font-bold mb-6">
          Route Performance Comparison
        </h1>

        {/* KPI SUMMARY */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-10">
          <div className="glass-card p-6">
            <p className="text-muted-foreground text-sm">Fuel Saved</p>
            <p className="text-2xl font-bold text-green-500">
              {data.fuelSaved?.toFixed(2) ?? 0}%
            </p>
          </div>

          <div className="glass-card p-6">
            <p className="text-muted-foreground text-sm">CO₂ Reduced</p>
            <p className="text-2xl font-bold text-green-500">
              {data.co2Reduction?.toFixed(2) ?? 0} t
            </p>
          </div>

          <div className="glass-card p-6">
            <p className="text-muted-foreground text-sm">Fuel Cost Saved</p>
            <p className="text-2xl font-bold text-green-500">
              ${data.fuelCostSavings?.toFixed(0) ?? 0}
            </p>
          </div>

          <div className="glass-card p-6">
            <p className="text-muted-foreground text-sm">IMO CII Rating</p>
            <p className="text-2xl font-bold text-primary">
              {data.ciiRating ?? "-"}
            </p>
          </div>
        </div>

        {/* 📊 BAR CHART */}
        <div className="glass-card p-8">
          <h2 className="text-lg font-semibold mb-6">
            Distance vs Fuel Comparison
          </h2>

          <div style={{ width: "100%", height: 400 }}>
            <ResponsiveContainer>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="route" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="distance" fill="#3b82f6" name="Distance (nm)" />
                <Bar dataKey="fuel" fill="#22c55e" name="Fuel (tons)" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

      </main>
      <Footer />
    </div>
  );
};

export default Analytics;