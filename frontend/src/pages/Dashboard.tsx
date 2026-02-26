// import Navbar from "@/components/Navbar";
// import Footer from "@/components/Footer";
// import KPICards from "@/components/KPICards";
// import MapSection from "@/components/MapSection";
// import AISummaryPanel from "@/components/AISummaryPanel";
// import { useOptimization } from "@/context/OptimizationContext";
// const Dashboard = () => {
//   // Static demo data for the dashboard view
//   const demoData = {
//     fuelSaved: 12.4,
//     co2Reduction: 18.7,
//     timeSaved: 4.2,
//     distanceRerouted: 42,
//   };
// const { data } = useOptimization();
//   const demoSummary =
//     "Last fleet optimization reduced average fuel consumption by 12.4% across 8 active vessels. Wind corridor analysis identified 3 high-resistance zones, prompting reroutes through calmer waters with favorable currents.";

//   return (
//     <div className="min-h-screen ocean-gradient flex flex-col">
//       <Navbar />
//       <main className="flex-1 pt-20">
//         <section className="px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto pt-8 pb-4">
//           <h1 className="text-2xl sm:text-3xl font-bold text-foreground mb-1">Dashboard</h1>
//           <p className="text-sm text-muted-foreground">Fleet performance overview</p>
//         </section>
//         <KPICards data={data} visible={!!data} />
//         <MapSection
//         baselineRoute={data?.baselineRoute}
//         optimizedRoute={data?.optimizedRoute}
//         visible={!!data}
//       />
//         <AISummaryPanel
//   summary={
//     data
//       ? `Baseline Distance: ${data.baselineDistance} nm | Optimized Distance: ${data.optimizedDistance} nm`
//       : null
//   }
//   visible={!!data}
// />
//       </main>
//       <Footer />
//     </div>
//   );
// };

// export default Dashboard;
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import KPICards from "@/components/KPICards";
import MapSection from "@/components/MapSection";
import AISummaryPanel from "@/components/AISummaryPanel";
import { useOptimization } from "@/context/OptimizationContext";
const Dashboard = () => {
  // Static demo data for the dashboard view
  const demoData = {
    fuelSaved: 12.4,
    co2Reduction: 18.7,
    timeSaved: 4.2,
    distanceRerouted: 42,
  };
const { data } = useOptimization();
  const demoSummary =
    "Last fleet optimization reduced average fuel consumption by 12.4% across 8 active vessels. Wind corridor analysis identified 3 high-resistance zones, prompting reroutes through calmer waters with favorable currents.";

  return (
    <div className="min-h-screen ocean-gradient flex flex-col">
      <Navbar />
      <main className="flex-1 pt-20">
        <section className="px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto pt-8 pb-4">
          <h1 className="text-2xl sm:text-3xl font-bold text-foreground mb-1">Dashboard</h1>
          <p className="text-sm text-muted-foreground">Fleet performance overview</p>
        </section>
        <KPICards data={data} visible={!!data} />
        <MapSection
        baselineRoute={data?.baselineRoute}
        optimizedRoute={data?.optimizedRoute}
        visible={!!data}
      />
        <AISummaryPanel
  summary={
    data
      ? `Baseline Distance: ${data.baselineDistance} nm | Optimized Distance: ${data.optimizedDistance} nm`
      : null
  }
  visible={!!data}
/>
      </main>
      <Footer />
    </div>
  );
};

export default Dashboard;
