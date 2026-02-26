// import { useState, useCallback } from "react";
// import Navbar from "@/components/Navbar";
// import Footer from "@/components/Footer";
// import HeroSection, { RouteInput } from "@/components/HeroSection";
// import KPICards from "@/components/KPICards";
// import MapSection from "@/components/MapSection";
// import AISummaryPanel from "@/components/AISummaryPanel";

// interface OptimizationResult {
//   fuelSaved: number;
//   co2Reduction: number;
//   timeSaved: number;
//   distanceRerouted: number;
//   summary: string;
// }

// const Index = () => {
//   const [result, setResult] = useState<OptimizationResult | null>(null);
//   const [isOptimizing, setIsOptimizing] = useState(false);
//   const [coords, setCoords] = useState<{
//     start: { lat: number; lon: number };
//     end: { lat: number; lon: number };
//   } | null>(null);

//   const handleOptimize = useCallback(async (input: RouteInput) => {
//     setIsOptimizing(true);
//     setResult(null);

//     const start = { lat: parseFloat(input.startLat), lon: parseFloat(input.startLon) };
//     const end = { lat: parseFloat(input.endLat), lon: parseFloat(input.endLon) };
//     setCoords({ start, end });

//     // Prepare payload for future API integration
//     const _payload = {
//       voyage: {
//         start_port: { lat: start.lat, lon: start.lon },
//         end_port: { lat: end.lat, lon: end.lon },
//       },
//     };

//     // Simulate API call — replace with POST /optimize
//     await new Promise((resolve) => setTimeout(resolve, 2200));

//     // Simulated result — replace with API response
//     const latDiff = Math.abs(start.lat - end.lat);
//     const lonDiff = Math.abs(start.lon - end.lon);
//     const distance = Math.sqrt(latDiff ** 2 + lonDiff ** 2) * 60;

//     setResult({
//       fuelSaved: 8.2 + Math.random() * 6,
//       co2Reduction: 12.5 + distance * 0.15,
//       timeSaved: 2.4 + Math.random() * 3,
//       distanceRerouted: Math.round(distance * 0.12 + Math.random() * 20),
//       summary: `AI rerouted the vessel from ${start.lat.toFixed(2)}°N, ${start.lon.toFixed(2)}°E to ${end.lat.toFixed(2)}°N, ${end.lon.toFixed(2)}°E, avoiding a high-pressure weather system. The optimized path reduces fuel consumption by leveraging favorable ocean currents while maintaining arrival schedule stability. Wind resistance along the original corridor was 23% above threshold, prompting a ${Math.round(distance * 0.12)}nm diversion through calmer waters.`,
//     });

//     setIsOptimizing(false);
//   }, []);

//   return (
//     <div className="min-h-screen ocean-gradient flex flex-col">
//       <Navbar />
//       <main className="flex-1">
//         <HeroSection onOptimize={handleOptimize} isOptimizing={isOptimizing} />
//         <div className="section-divider max-w-7xl mx-auto" />
//         <KPICards
//           data={result ? {
//             fuelSaved: result.fuelSaved,
//             co2Reduction: result.co2Reduction,
//             timeSaved: result.timeSaved,
//             distanceRerouted: result.distanceRerouted,
//           } : null}
//           visible={!!result}
//         />
//         <MapSection
//           startCoords={coords?.start ?? null}
//           endCoords={coords?.end ?? null}
//           visible={!!result}
//         />
//         <AISummaryPanel summary={result?.summary ?? null} visible={!!result} />
//       </main>
//       <Footer />
//     </div>
//   );
// };

// export default Index;


// import { useState, useCallback } from "react";
// import axios from "axios";
// import Navbar from "@/components/Navbar";
// import Footer from "@/components/Footer";
// import HeroSection, { RouteInput } from "@/components/HeroSection";
// import KPICards from "@/components/KPICards";
// import MapSection from "@/components/MapSection";
// import AISummaryPanel from "@/components/AISummaryPanel";

// interface OptimizationResult {
//   fuelSaved: number;
//   co2Reduction: number;
//   timeSaved: number;
//   distanceRerouted: number;
//   summary: string;
// }

// const Index = () => {
//   const [result, setResult] = useState<OptimizationResult | null>(null);
//   const [isOptimizing, setIsOptimizing] = useState(false);
//   const [coords, setCoords] = useState<{
//     start: { lat: number; lon: number };
//     end: { lat: number; lon: number };
//   } | null>(null);

//   const handleOptimize = useCallback(async (input: RouteInput) => {
//     try {
//       setIsOptimizing(true);
//       setResult(null);

//       const start = {
//         lat: parseFloat(input.startLat),
//         lon: parseFloat(input.startLon),
//       };

//       const end = {
//         lat: parseFloat(input.endLat),
//         lon: parseFloat(input.endLon),
//       };

//       setCoords({ start, end });

//       const response = await axios.post("http://localhost:8000/optimize", {
//         voyage: {
//           start_port: { lat: start.lat, lon: start.lon },
//           end_port: { lat: end.lat, lon: end.lon },
//         },
//       });

//       const data = response.data;

//       setResult({
//         fuelSaved: parseFloat(
//           data["Optimized Route Generated"].match(/([\d.]+)%/)?.[1] || "0"
//         ),
//         co2Reduction: parseFloat(
//           data["Emissions Report"].match(/([\d.]+)/)?.[1] || "0"
//         ),
//         timeSaved: parseFloat(
//           data["Operational Insight"].match(/±([\d.]+)/)?.[1] || "0"
//         ),
//         distanceRerouted: parseFloat(
//           data["Weather Impact Adjustment"].match(/([\d.]+) nautical/)?.[1] ||
//             "0"
//         ),
//         summary:
//           data["Optimized Route Generated"] +
//           " " +
//           data["Weather Impact Adjustment"] +
//           " " +
//           data["Emissions Report"] +
//           " " +
//           data["Operational Insight"],
//       });
//     } catch (error) {
//       console.error("Optimization failed:", error);
//       alert("Backend connection failed. Make sure FastAPI is running.");
//     } finally {
//       setIsOptimizing(false);
//     }
//   }, []);

//   return (
//     <div className="min-h-screen ocean-gradient flex flex-col">
//       <Navbar />
//       <main className="flex-1">
//         <HeroSection onOptimize={handleOptimize} isOptimizing={isOptimizing} />
//         <div className="section-divider max-w-7xl mx-auto" />
//         <KPICards
//           data={
//             result
//               ? {
//                   fuelSaved: result.fuelSaved,
//                   co2Reduction: result.co2Reduction,
//                   timeSaved: result.timeSaved,
//                   distanceRerouted: result.distanceRerouted,
//                 }
//               : null
//           }
//           visible={!!result}
//         />
//         <MapSection
//           startCoords={coords?.start ?? null}
//           endCoords={coords?.end ?? null}
//           visible={!!result}
//         />
//         <AISummaryPanel summary={result?.summary ?? null} visible={!!result} />
//       </main>
//       <Footer />
//     </div>
//   );
// };

// export default Index;






// import { useState, useCallback } from "react";
// import axios from "axios";
// import Navbar from "@/components/Navbar";
// import Footer from "@/components/Footer";
// import HeroSection, { RouteInput } from "@/components/HeroSection";
// import KPICards from "@/components/KPICards";
// import MapSection from "@/components/MapSection";
// import AISummaryPanel from "@/components/AISummaryPanel";
// import { useOptimization } from "@/context/OptimizationContext";
// interface OptimizationResult {
//   fuelSaved: number;
//   co2Reduction: number;
//   timeSaved: number;
//   distanceRerouted: number;
//   summary: string;
// }

// const Index = () => {
//   const [result, setResult] = useState<OptimizationResult | null>(null);
//   const [isOptimizing, setIsOptimizing] = useState(false);
//   const { setData } = useOptimization();
//   const [coords, setCoords] = useState<{
//     start: { lat: number; lon: number };
//     end: { lat: number; lon: number };
//   } | null>(null);

//   // const handleOptimize = useCallback(async (input: RouteInput) => {
//   //   try {
//   //     setIsOptimizing(true);
//   //     setResult(null);

//   //     const start = {
//   //       lat: parseFloat(input.startLat),
//   //       lon: parseFloat(input.startLon),
//   //     };

//   //     const end = {
//   //       lat: parseFloat(input.endLat),
//   //       lon: parseFloat(input.endLon),
//   //     };

//   //     setCoords({ start, end });

//   //     // const response = await axios.post("http://localhost:8000/optimize", {
//   //     axios.post(`${import.meta.env.VITE_API_URL}/optimize`, {
//   //       voyage: {
//   //         start_port: { lat: start.lat, lon: start.lon },
//   //         end_port: { lat: end.lat, lon: end.lon },
//   //       },
//   //     });

//   //     const data = response.data;

//   //     setResult({
//   //       fuelSaved: data.fuel_reduction_percent,
//   //       co2Reduction: data.co2_reduction_tons,
//   //       timeSaved: data.time_difference_hours,
//   //       distanceRerouted: data.rerouted_distance_nm,
//   //       summary: `Baseline Distance: ${data.baseline_distance_nm} nm | Optimized Distance: ${data.optimized_distance_nm} nm`,
//   //     });
//   //   } catch (error) {
//   //     console.error(error);
//   //     alert("Backend connection failed.");
//   //   } finally {
//   //     setIsOptimizing(false);
//   //   }
//   // }, []);
//   const handleOptimize = useCallback(async (input: RouteInput) => {
//   try {
//     setIsOptimizing(true);
//     setResult(null);

//     const start = {
//       lat: parseFloat(input.startLat),
//       lon: parseFloat(input.startLon),
//     };

//     const end = {
//       lat: parseFloat(input.endLat),
//       lon: parseFloat(input.endLon),
//     };

//     setCoords({ start, end });

//     const response = await axios.post(
//       `${import.meta.env.VITE_API_URL}/optimize`,
//       {
//         voyage: {
//           start_port: { lat: start.lat, lon: start.lon },
//           end_port: { lat: end.lat, lon: end.lon },
//         },
//       }
//     );

//     const data = response.data;

//     setData({
//   fuelSaved: data.fuel_reduction_percent,
//   co2Reduction: data.co2_reduction_tons,
//   timeSaved: data.time_difference_hours,
//   distanceRerouted: data.rerouted_distance_nm,
//   baselineDistance: data.baseline_distance_nm,
//   optimizedDistance: data.optimized_distance_nm,
//   start,
//   end,
// });
//   } catch (error) {
//     console.error(error);
//     alert("Backend connection failed.");
//   } finally {
//     setIsOptimizing(false);
//   }
// }, []);

//   return (
//     <div className="min-h-screen ocean-gradient flex flex-col">
//       <Navbar />
//       <main className="flex-1">
//         <HeroSection onOptimize={handleOptimize} isOptimizing={isOptimizing} />
//         <div className="section-divider max-w-7xl mx-auto" />
//         <KPICards
//           data={result ? {
//             fuelSaved: result.fuelSaved,
//             co2Reduction: result.co2Reduction,
//             timeSaved: result.timeSaved,
//             distanceRerouted: result.distanceRerouted,
//           } : null}
//           visible={!!result}
//         />
//         <MapSection
//           startCoords={coords?.start ?? null}
//           endCoords={coords?.end ?? null}
//           visible={!!result}
//         />
//         <AISummaryPanel summary={result?.summary ?? null} visible={!!result} />
//       </main>
//       <Footer />
//     </div>
//   );
// };

// export default Index;
// import { useState, useCallback } from "react";
// import axios from "axios";
// import Navbar from "@/components/Navbar";
// import Footer from "@/components/Footer";
// import HeroSection, { RouteInput } from "@/components/HeroSection";
// import KPICards from "@/components/KPICards";
// import MapSection from "@/components/MapSection";
// import AISummaryPanel from "@/components/AISummaryPanel";
// import { useOptimization } from "@/context/OptimizationContext";

// const Index = () => {
//   const [isOptimizing, setIsOptimizing] = useState(false);
//   const { data, setData } = useOptimization();

//   const handleOptimize = useCallback(async (input: RouteInput) => {
//     try {
//       setIsOptimizing(true);

//       const start = {
//         lat: parseFloat(input.startLat),
//         lon: parseFloat(input.startLon),
//       };

//       const end = {
//         lat: parseFloat(input.endLat),
//         lon: parseFloat(input.endLon),
//       };

//       const response = await axios.post(
//         `${import.meta.env.VITE_API_URL}/optimize`,
//         {
//           voyage: {
//             start_port: start,
//             end_port: end,
//           },
//         }
//       );

//       const res = response.data;

//       setData({
//         fuelSaved: res.fuel_reduction_percent,
//         co2Reduction: res.co2_reduction_tons,
//         timeSaved: res.time_difference_hours,
//         distanceRerouted: res.rerouted_distance_nm,
//         baselineDistance: res.baseline_distance_nm,
//         optimizedDistance: res.optimized_distance_nm,
//         start,
//         end,
//       });

//     } catch (error) {
//       console.error(error);
//       alert("Backend connection failed.");
//     } finally {
//       setIsOptimizing(false);
//     }
//   }, []);

//   return (
//     <div className="min-h-screen ocean-gradient flex flex-col">
//       <Navbar />
//       <main className="flex-1">
//         <HeroSection onOptimize={handleOptimize} isOptimizing={isOptimizing} />

//         <div className="section-divider max-w-7xl mx-auto" />

//         <KPICards
//           data={data}
//           visible={!!data}
//         />

//         <MapSection
//           startCoords={data?.start ?? null}
//           endCoords={data?.end ?? null}
//           visible={!!data}
//         />

//         <AISummaryPanel
//           summary={
//             data
//               ? `Baseline Distance: ${data.baselineDistance} nm | Optimized Distance: ${data.optimizedDistance} nm`
//               : null
//           }
//           visible={!!data}
//         />
//       </main>
//       <Footer />
//     </div>
//   );
// };

// export default Index;
// import { useState, useCallback, useEffect } from "react";
// import axios from "axios";
// import Navbar from "@/components/Navbar";
// import Footer from "@/components/Footer";
// import HeroSection, { RouteInput } from "@/components/HeroSection";
// import KPICards from "@/components/KPICards";
// import MapSection from "@/components/MapSection";
// import AISummaryPanel from "@/components/AISummaryPanel";
// import { useOptimization } from "@/context/OptimizationContext";

// const Index = () => {
//   const [isOptimizing, setIsOptimizing] = useState(false);
//   const { data, setData } = useOptimization();

//   // ✅ OPTION 2: Clear stored optimization ONLY when Home loads
//   useEffect(() => {
//     localStorage.removeItem("optimizationData");
//   }, []);

//   const handleOptimize = useCallback(
//     async (input: RouteInput) => {
//       try {
//         setIsOptimizing(true);

//         const start = {
//           lat: parseFloat(input.startLat),
//           lon: parseFloat(input.startLon),
//         };

//         const end = {
//           lat: parseFloat(input.endLat),
//           lon: parseFloat(input.endLon),
//         };

//         const response = await axios.post(
//           `${import.meta.env.VITE_API_URL}/optimize`,
//           {
//             voyage: {
//               start_port: start,
//               end_port: end,
//             },
//           }
//         );

//         const res = response.data;

//         setData({
//           fuelSaved: res.fuel_reduction_percent,
//           co2Reduction: res.co2_reduction_tons,
//           timeSaved: res.time_difference_hours,
//           distanceRerouted: res.rerouted_distance_nm,
//           baselineDistance: res.baseline_distance_nm,
//           optimizedDistance: res.optimized_distance_nm,
//           start,
//           end,
//         });
//       } catch (error) {
//         console.error(error);
//         alert("Backend connection failed.");
//       } finally {
//         setIsOptimizing(false);
//       }
//     },
//     [setData]
//   );

//   return (
//     <div className="min-h-screen ocean-gradient flex flex-col">
//       <Navbar />
//       <main className="flex-1">
//         <HeroSection onOptimize={handleOptimize} isOptimizing={isOptimizing} />

//         <div className="section-divider max-w-7xl mx-auto" />

//         <KPICards data={data} visible={!!data} />

//         <MapSection
//           startCoords={data?.start ?? null}
//           endCoords={data?.end ?? null}
//           visible={!!data}
//         />

//         <AISummaryPanel
//           summary={
//             data
//               ? `Baseline Distance: ${data.baselineDistance} nm | Optimized Distance: ${data.optimizedDistance} nm`
//               : null
//           }
//           visible={!!data}
//         />
//       </main>
//       <Footer />
//     </div>
//   );
// };

// export default Index;
// import { useState, useCallback } from "react";
// import axios from "axios";
// import Navbar from "@/components/Navbar";
// import Footer from "@/components/Footer";
// import HeroSection, { RouteInput } from "@/components/HeroSection";
// import KPICards from "@/components/KPICards";
// import MapSection from "@/components/MapSection";
// import AISummaryPanel from "@/components/AISummaryPanel";
// import { useOptimization } from "@/context/OptimizationContext";

// const Index = () => {
//   const [isOptimizing, setIsOptimizing] = useState(false);
//   const { data, setData } = useOptimization();

//   const handleOptimize = useCallback(
//     async (input: RouteInput) => {
//       try {
//         setIsOptimizing(true);

//         const start = {
//           lat: parseFloat(input.startLat),
//           lon: parseFloat(input.startLon),
//         };

//         const end = {
//           lat: parseFloat(input.endLat),
//           lon: parseFloat(input.endLon),
//         };

//         const response = await axios.post(
//           `${import.meta.env.VITE_API_URL}/optimize`,
//           {
//             voyage: {
//               start_port: start,
//               end_port: end,
//             },
//           }
//         );

//         const res = response.data;

//         // 🔥 Correct mapping for NEW backend keys
//         setData({
//           fuelSaved: res["Fuel Reduction (%)"],
//           co2Reduction: res["CO2 Reduction (tons)"],
//           timeSaved: res["Time Difference (hours)"],
//           distanceRerouted: res["Route Adjustment (NM)"],
//           baselineDistance: res["Baseline Distance (NM)"],
//           optimizedDistance: res["Optimized Distance (NM)"],
//           start,
//           end,
//         });

//       } catch (error) {
//         console.error(error);
//         alert("Backend connection failed.");
//       } finally {
//         setIsOptimizing(false);
//       }
//     },
//     [setData]
//   );

//   return (
//     <div className="min-h-screen ocean-gradient flex flex-col">
//       <Navbar />
//       <main className="flex-1">
//         <HeroSection onOptimize={handleOptimize} isOptimizing={isOptimizing} />

//         <div className="section-divider max-w-7xl mx-auto" />

//         <KPICards data={data} visible={!!data} />

//         <MapSection
//           startCoords={data?.start ?? null}
//           endCoords={data?.end ?? null}
//           visible={!!data}
//         />

//         <AISummaryPanel
//           summary={
//             data
//               ? `Baseline Distance: ${data.baselineDistance} nm | Optimized Distance: ${data.optimizedDistance} nm`
//               : null
//           }
//           visible={!!data}
//         />
//       </main>
//       <Footer />
//     </div>
//   );
// };

// export default Index;
// import { useState, useCallback } from "react";
// import axios from "axios";
// import Navbar from "@/components/Navbar";
// import Footer from "@/components/Footer";
// import HeroSection, { RouteInput } from "@/components/HeroSection";
// import KPICards from "@/components/KPICards";
// import MapSection from "@/components/MapSection";
// import AISummaryPanel from "@/components/AISummaryPanel";
// import { useOptimization } from "@/context/OptimizationContext";

// const Index = () => {
//   const [isOptimizing, setIsOptimizing] = useState(false);
//   const { data, setData } = useOptimization();

//   const handleOptimize = useCallback(
//     async (input: RouteInput) => {
//       try {
//         setIsOptimizing(true);

//         const start = {
//           lat: parseFloat(input.startLat),
//           lon: parseFloat(input.startLon),
//         };

//         const end = {
//           lat: parseFloat(input.endLat),
//           lon: parseFloat(input.endLon),
//         };

//         const response = await axios.post(
//           `${import.meta.env.VITE_API_URL}/optimize`,
//           {
//             voyage: {
//               start_port: start,
//               end_port: end,
//             },
//           }
//         );

//         const res = response.data;

//         // ✅ CORRECT mapping for snake_case backend
//         setData({
//           fuelSaved: res.fuel_reduction_percent,
//           co2Reduction: res.co2_reduction_tons,
//           timeSaved: res.time_difference_hours,
//           distanceRerouted: res.rerouted_distance_nm,
//           baselineDistance: res.baseline_distance_nm,
//           optimizedDistance: res.optimized_distance_nm,
//           baselineRoute: res.baseline_route,
//           optimizedRoute: res.optimized_route,
//         });

//       } catch (error) {
//         console.error(error);
//         alert("Backend connection failed.");
//       } finally {
//         setIsOptimizing(false);
//       }
//     },
//     [setData]
//   );

//   return (
//     <div className="min-h-screen ocean-gradient flex flex-col">
//       <Navbar />
//       <main className="flex-1">
//         <HeroSection onOptimize={handleOptimize} isOptimizing={isOptimizing} />

//         <div className="section-divider max-w-7xl mx-auto" />

//         <KPICards data={data} visible={!!data} />

//         <MapSection
//         baselineRoute={data?.baselineRoute}
//         optimizedRoute={data?.optimizedRoute}
//         visible={!!data}
//       />
//         <AISummaryPanel
//           summary={
//             data
//               ? `Baseline Distance: ${data.baselineDistance} nm | Optimized Distance: ${data.optimizedDistance} nm`
//               : null
//           }
//           visible={!!data}
//         />
//       </main>
//       <Footer />
//     </div>
//   );
// };

// export default Index;
// import { useState, useCallback } from "react";
// import axios from "axios";
// import Navbar from "@/components/Navbar";
// import Footer from "@/components/Footer";
// import HeroSection, { RouteInput } from "@/components/HeroSection";
// import KPICards from "@/components/KPICards";
// import MapSection from "@/components/MapSection";
// import AISummaryPanel from "@/components/AISummaryPanel";
// import { useOptimization } from "@/context/OptimizationContext";

// const Index = () => {
//   const [isOptimizing, setIsOptimizing] = useState(false);
//   const { data, setData } = useOptimization();

//   const handleOptimize = useCallback(
//     async (input: RouteInput) => {
//       try {
//         setIsOptimizing(true);

//         const response = await axios.post(
//           `${import.meta.env.VITE_API_URL}/optimize`,
//           {
//             voyage: {
//               start_port: {
//                 lat: Number(input.startLat),
//                 lon: Number(input.startLon),
//               },
//               end_port: {
//                 lat: Number(input.endLat),
//                 lon: Number(input.endLon),
//               },
//             },
//             vessel: {
//               vessel_type: input.vesselType || "container",
//               cargo_tons: Number(input.cargoTons) || 30000,
//               engine_power_kw: Number(input.enginePowerKw) || 15000,
//             },
//           }
//         );

//         const res = response.data;

//         setData({
//           fuelSaved: res.fuel_reduction_percent,
//           co2Reduction: res.co2_reduction_tons,
//           baselineDistance: res.baseline_distance_nm,
//           optimizedDistance: res.optimized_distance_nm,
//           baselineRoute: res.baseline_route,
//           optimizedRoute: res.optimized_route,
//           selectedRoute: res.selected_route,
//           timestamp: res.timestamp,
//           fuelReductionPercent: res.fuel_reduction_percent,
//           co2ReductionTons: res.co2_reduction_tons,
//         });

//       } catch (error) {
//         console.error("Optimization error:", error);
//         alert("Backend connection failed.");
//       } finally {
//         setIsOptimizing(false);
//       }
//     },
//     [setData]
//   );

//   return (
//     <div className="min-h-screen ocean-gradient flex flex-col">
//       <Navbar />
//       <main className="flex-1">
//         <HeroSection onOptimize={handleOptimize} isOptimizing={isOptimizing} />

//         <div className="section-divider max-w-7xl mx-auto" />

//         <KPICards data={data} visible={!!data} />

//         <MapSection
//           baselineRoute={data?.baselineRoute}
//           optimizedRoute={data?.optimizedRoute}
//           visible={!!data}
//         />

//         <AISummaryPanel
//           summary={
//             data
//               ? `Baseline Distance: ${data.baselineDistance} nm | Optimized Distance: ${data.optimizedDistance} nm`
//               : null
//           }
//           visible={!!data}
//         />
//       </main>
//       <Footer />
//     </div>
//   );
// };

// export default Index;
// import { useState, useCallback } from "react";
// import axios from "axios";
// import Navbar from "@/components/Navbar";
// import Footer from "@/components/Footer";
// import HeroSection, { RouteInput } from "@/components/HeroSection";
// import KPICards from "@/components/KPICards";
// import MapSection from "@/components/MapSection";
// import AISummaryPanel from "@/components/AISummaryPanel";
// import { useOptimization } from "@/context/OptimizationContext";

// const Index = () => {
//   const [isOptimizing, setIsOptimizing] = useState(false);
//   const { data, setData } = useOptimization();

//   const handleOptimize = useCallback(
//     async (input: RouteInput) => {
//       try {
//         setIsOptimizing(true);

//         const response = await axios.post(
//           `${import.meta.env.VITE_API_URL}/optimize`,
//           {
//             voyage: {
//               start_port: {
//                 lat: Number(input.startLat),
//                 lon: Number(input.startLon),
//               },
//               end_port: {
//                 lat: Number(input.endLat),
//                 lon: Number(input.endLon),
//               },
//             },
//             vessel: {
//               vessel_type: "container",
//               cargo_tons: 30000,
//               engine_power_kw: 15000,
//             },
//           }
//         );

//         const res = response.data;

// //         setData({
// //   fuelSaved: res.fuel_reduction_percent,
// //   co2Reduction: res.co2_reduction_tons,
// //   fuelCostSavings: res.fuel_cost_savings_usd,
// //   ciiRating: res.cii_rating,
// //   baselineDistance: res.baseline_distance_nm,
// //   optimizedDistance: res.optimized_distance_nm,
// //   baselineRoute: res.baseline_route,
// //   optimizedRoute: res.optimized_route,
// //   routeComparison: res.route_comparison,
// //   timestamp: res.timestamp,
// // });
// setData({
//   fuelSaved: res.fuel_reduction_percent ?? 0,
//   co2Reduction: res.co2_reduction_tons ?? 0,
//   timeSaved: res.time_difference_hours ?? 0,
//   distanceRerouted: res.rerouted_distance_nm ?? 0,
//   fuelCostSavings: res.fuel_cost_savings_usd ?? 0,
//   ciiRating: res.cii_rating ?? "C",
//   baselineDistance: res.baseline_distance_nm ?? 0,
//   optimizedDistance: res.optimized_distance_nm ?? 0,
//   baselineRoute: res.baseline_route ?? [],
//   optimizedRoute: res.optimized_route ?? [],
//   routeComparison: res.route_comparison ?? {},
//   timestamp: res.timestamp ?? "",
// });
//       } catch (error) {
//         console.error("Optimization error:", error);
//         alert("Backend connection failed.");
//       } finally {
//         setIsOptimizing(false);
//       }
//     },
//     [setData]
//   );
//   const downloadReport = async () => {
//   const response = await axios.post(
//     `${import.meta.env.VITE_API_URL}/generate-report`,
//     data,
//     { responseType: "blob" }
//   );

//   const url = window.URL.createObjectURL(new Blob([response.data]));
//   const link = document.createElement("a");
//   link.href = url;
//   link.setAttribute("download", "NaviGreen_Report.pdf");
//   document.body.appendChild(link);
//   link.click();
// };

//   return (
//     <div className="min-h-screen ocean-gradient flex flex-col">
//       <Navbar />
//       <main className="flex-1">
//         <HeroSection onOptimize={handleOptimize} isOptimizing={isOptimizing} />

//         <div className="section-divider max-w-7xl mx-auto" />

//         <KPICards data={data} visible={!!data} />
// {data?.routeComparison && (
//   <div className="glass-card p-6 max-w-7xl mx-auto mt-6">
//     <h3 className="text-lg font-semibold mb-4">Route Comparison</h3>
//     <table className="w-full text-sm">
//       <thead>
//         <tr>
//           <th>Route</th>
//           <th>Distance (nm)</th>
//           <th>Fuel (tons)</th>
//         </tr>
//       </thead>
//       <tbody>
//         {Object.entries(data.routeComparison).map(([key,val]:any)=>(
//           <tr key={key}>
//             <td>{key}</td>
//             <td>{val.distance}</td>
//             <td>{val.fuel}</td>
//           </tr>
//         ))}
//       </tbody>
//     </table>
//   </div>
// )}
//         <MapSection
//           baselineRoute={data?.baselineRoute}
//           optimizedRoute={data?.optimizedRoute}
//           visible={!!data}
//         />

//         <AISummaryPanel
//           summary={
//             data
//               ? `Baseline Distance: ${data.baselineDistance} nm | Optimized Distance: ${data.optimizedDistance} nm`
//               : null
//           }
//           visible={!!data}
//         />
//       </main>
//       <Footer />
//     </div>
//   );
// };

// export default Index;

// import { useState, useCallback } from "react";
// import axios from "axios";
// import Navbar from "@/components/Navbar";
// import Footer from "@/components/Footer";
// import HeroSection, { RouteInput } from "@/components/HeroSection";
// import KPICards from "@/components/KPICards";
// import MapSection from "@/components/MapSection";
// import AISummaryPanel from "@/components/AISummaryPanel";
// import { useOptimization } from "@/context/OptimizationContext";

// const Index = () => {
//   const [isOptimizing, setIsOptimizing] = useState(false);
//   const { data, setData } = useOptimization();

//   const handleOptimize = useCallback(
//     async (input: RouteInput) => {
//       try {
//         setIsOptimizing(true);

//         const response = await axios.post(
//           `${import.meta.env.VITE_API_URL}/optimize`,
//           {
//             voyage: {
//               start_port: {
//                 lat: Number(input.startLat),
//                 lon: Number(input.startLon),
//               },
//               end_port: {
//                 lat: Number(input.endLat),
//                 lon: Number(input.endLon),
//               },
//             },
//           }
//         );

//         const res = response.data;

//         setData({
//           fuelSaved: res.fuel_reduction_percent ?? 0,
//           co2Reduction: res.co2_reduction_tons ?? 0,
//           timeSaved: 0, // backend not returning this
//           distanceRerouted: 0, // backend not returning this
//           fuelCostSavings: res.fuel_cost_savings_usd ?? 0,
//           ciiRating: res.cii_rating ?? "C",
//           baselineDistance: res.baseline_distance_nm ?? 0,
//           optimizedDistance: res.optimized_distance_nm ?? 0,
//           baselineRoute: res.baseline_route ?? [],
//           optimizedRoute: res.optimized_route ?? [],
//           routeComparison: res.route_comparison ?? {},
//           timestamp: res.timestamp ?? "",
//           weatherSamples: res.weather_samples ?? [],
//         });
//       } catch (error) {
//         console.error("Optimization error:", error);
//         alert("Backend connection failed.");
//       } finally {
//         setIsOptimizing(false);
//       }
//     },
//     [setData]
//   );

//   const downloadReport = async () => {
//     if (!data) return;

//     const response = await axios.post(
//       `${import.meta.env.VITE_API_URL}/generate-report`,
//       data,
//       { responseType: "blob" }
//     );

//     const url = window.URL.createObjectURL(new Blob([response.data]));
//     const link = document.createElement("a");
//     link.href = url;
//     link.setAttribute("download", "NaviGreen_Report.pdf");
//     document.body.appendChild(link);
//     link.click();
//   };

//   return (
//     <div className="min-h-screen ocean-gradient flex flex-col">
//       <Navbar />
//       <main className="flex-1">
//         <HeroSection onOptimize={handleOptimize} isOptimizing={isOptimizing} />

//         <div className="section-divider max-w-7xl mx-auto" />

//         <KPICards data={data} visible={!!data} />

//         {/* ✅ DOWNLOAD BUTTON */}
//         {data && (
//           <div className="text-center mt-6">
//             <button
//               onClick={downloadReport}
//               className="bg-primary text-white px-6 py-3 rounded-xl shadow-lg hover:scale-105 transition"
//             >
//               📄 Download Optimization Report
//             </button>
//           </div>
//         )}

//         {/* ✅ ROUTE COMPARISON TABLE */}
//         {data?.routeComparison && (
//           <div className="glass-card p-6 max-w-7xl mx-auto mt-6">
//             <h3 className="text-lg font-semibold mb-4">
//               Route Comparison
//             </h3>
//             <table className="w-full text-sm">
//               <thead>
//                 <tr>
//                   <th>Route</th>
//                   <th>Distance (nm)</th>
//                   <th>Fuel (tons)</th>
//                 </tr>
//               </thead>
//               <tbody>
//                 {Object.entries(data.routeComparison).map(
//                   ([key, val]: any) => (
//                     <tr key={key}>
//                       <td>{key}</td>
//                       <td>{val.distance}</td>
//                       <td>{val.fuel}</td>
//                     </tr>
//                   )
//                 )}
//               </tbody>
//             </table>
//           </div>
//         )}

//         <MapSection
//   baselineRoute={data?.baselineRoute}
//   optimizedRoute={data?.optimizedRoute}
//   weatherSamples={data?.weatherSamples}
//   visible={!!data}
// />

//         <AISummaryPanel
//           summary={
//             data
//               ? `Baseline Distance: ${data.baselineDistance} nm | Optimized Distance: ${data.optimizedDistance} nm`
//               : null
//           }
//           visible={!!data}
//         />
//       </main>
//       <Footer />
//     </div>
//   );
// };

// export default Index;



// import { useState, useCallback } from "react";
// import axios from "axios";
// import Navbar from "@/components/Navbar";
// import Footer from "@/components/Footer";
// import HeroSection, { RouteInput } from "@/components/HeroSection";
// import KPICards from "@/components/KPICards";
// import MapSection from "@/components/MapSection";
// import AISummaryPanel from "@/components/AISummaryPanel";
// import { useOptimization } from "@/context/OptimizationContext";

// const Index = () => {
//   const [isOptimizing, setIsOptimizing] = useState(false);
//   const { data, setData } = useOptimization();

//   const handleOptimize = useCallback(
//     async (input: RouteInput) => {
//       try {
//         setIsOptimizing(true);

//         const response = await axios.post(
//           `${import.meta.env.VITE_API_URL}/optimize`,
//           {
//             voyage: {
//               start_port: {
//                 lat: Number(input.startLat),
//                 lon: Number(input.startLon),
//               },
//               end_port: {
//                 lat: Number(input.endLat),
//                 lon: Number(input.endLon),
//               },
//             },
//           }
//         );

//         const res = response.data;

//         setData({
//           fuelSaved: res.fuel_reduction_percent ?? 0,
//           co2Reduction: res.co2_reduction_tons ?? 0,
//           timeSaved: res.time_saved_hours ?? 0,
//           distanceRerouted:
//             Math.abs(
//               (res.baseline_distance_nm ?? 0) -
//                 (res.optimized_distance_nm ?? 0)
//             ) ?? 0,
//           fuelCostSavings: res.fuel_cost_savings_usd ?? 0,
//           ciiRating: res.cii_rating ?? "C",
//           baselineDistance: res.baseline_distance_nm ?? 0,
//           optimizedDistance: res.optimized_distance_nm ?? 0,
//           baselineRoute: res.baseline_route ?? [],
//           optimizedRoute: res.optimized_route ?? [],
//           routeComparison: res.route_comparison ?? {},
//           weatherSamples: res.weather_samples ?? [],
//           timestamp: res.timestamp ?? "",
//         });
//       } catch (error) {
//         console.error("Optimization error:", error);
//         alert("Backend connection failed.");
//       } finally {
//         setIsOptimizing(false);
//       }
//     },
//     [setData]
//   );

//   const downloadReport = async () => {
//     if (!data) return;

//     const response = await axios.post(
//       `${import.meta.env.VITE_API_URL}/generate-report`,
//       data,
//       { responseType: "blob" }
//     );

//     const url = window.URL.createObjectURL(new Blob([response.data]));
//     const link = document.createElement("a");
//     link.href = url;
//     link.setAttribute("download", "NaviGreen_Report.pdf");
//     document.body.appendChild(link);
//     link.click();
//   };

//   return (
//     <div className="min-h-screen ocean-gradient flex flex-col">
//       <Navbar />
//       <main className="flex-1">
//         <HeroSection
//           onOptimize={handleOptimize}
//           isOptimizing={isOptimizing}
//         />

//         <div className="section-divider max-w-7xl mx-auto" />

//         <KPICards data={data} visible={!!data} />

//         {/* 📄 Download Button */}
//         {data && (
//           <div className="text-center mt-6">
//             <button
//               onClick={downloadReport}
//               className="bg-primary text-white px-6 py-3 rounded-xl shadow-lg hover:scale-105 transition"
//             >
//               📄 Download Optimization Report
//             </button>
//           </div>
//         )}

//         {/* 📊 Route Comparison Table */}
//         {/* {data?.routeComparison &&
//           Object.keys(data.routeComparison).length > 0 && (
//             <div className="glass-card p-6 max-w-7xl mx-auto mt-6">
//               <h3 className="text-lg font-semibold mb-4">
//                 Route Comparison
//               </h3>
//               <table className="w-full text-sm">
//                 <thead>
//                   <tr>
//                     <th>Route</th>
//                     <th>Distance (nm)</th>
//                     <th>Fuel (tons)</th>
//                   </tr>
//                 </thead>
//                 <tbody>
//                   {Object.entries(data.routeComparison).map(
//                     ([key, val]: any) => (
//                       <tr key={key}>
//                         <td>{key}</td>
//                         <td>{val.distance}</td>
//                         <td>{val.fuel}</td>
//                       </tr>
//                     )
//                   )}
//                 </tbody>
//               </table>
//             </div>
//           )} */}
//           {data?.routeComparison && (
//   <div className="glass-card p-6 max-w-7xl mx-auto mt-6">
//     <h3 className="text-lg font-semibold mb-6 text-center">
//       Route Comparison
//     </h3>

//     <div className="overflow-x-auto">
//       <table className="w-full text-sm text-center">
//         <thead>
//           <tr className="border-b border-border/50">
//             <th className="py-3">Route</th>
//             <th className="py-3">Distance (nm)</th>
//             <th className="py-3">Fuel (tons)</th>
//           </tr>
//         </thead>

//         <tbody>
//           {Object.entries(data.routeComparison).map(
//             ([key, val]: any) => (
//               <tr
//                 key={key}
//                 className="border-b border-border/30 hover:bg-white/5 transition"
//               >
//                 <td className="py-3 font-medium">{key}</td>
//                 <td className="py-3">{val.distance}</td>
//                 <td className="py-3">{val.fuel}</td>
//               </tr>
//             )
//           )}
//         </tbody>
//       </table>
//     </div>
//   </div>
// )}

//         <MapSection
//           baselineRoute={data?.baselineRoute}
//           optimizedRoute={data?.optimizedRoute}
//           weatherSamples={data?.weatherSamples}
//           visible={!!data}
//         />

//         <AISummaryPanel
//           summary={
//             data
//               ? `Baseline Distance: ${data.baselineDistance} nm | Optimized Distance: ${data.optimizedDistance} nm`
//               : null
//           }
//           visible={!!data}
//         />
//       </main>
//       <Footer />
//     </div>
//   );
// };

// export default Index;
// import { useState, useCallback } from "react";
// import axios from "axios";
// import Navbar from "@/components/Navbar";
// import Footer from "@/components/Footer";
// import HeroSection, { RouteInput } from "@/components/HeroSection";
// import KPICards from "@/components/KPICards";
// import MapSection from "@/components/MapSection";
// import AISummaryPanel from "@/components/AISummaryPanel";
// import { useOptimization } from "@/context/OptimizationContext";

// const Index = () => {
//   const [isOptimizing, setIsOptimizing] = useState(false);
//   const { data, setData } = useOptimization();

//   const handleOptimize = useCallback(
//     async (input: RouteInput) => {
//       try {
//         setIsOptimizing(true);

//         const response = await axios.post(
//           `${import.meta.env.VITE_API_URL}/optimize`,
//           {
//             voyage: {
//               start_port: {
//                 lat: Number(input.startLat),
//                 lon: Number(input.startLon),
//               },
//               end_port: {
//                 lat: Number(input.endLat),
//                 lon: Number(input.endLon),
//               },
//             },
//           }
//         );

//         const res = response.data;

//         const baselineDistance =
//           res.baseline_distance_nm ?? 0;
//         const optimizedDistance =
//           res.optimized_distance_nm ?? 0;

//         setData({
//           selectedRoute: res.selected_route ?? "Direct",

//           fuelSaved: res.fuel_reduction_percent ?? 0,
//           co2Reduction: res.co2_reduction_tons ?? 0,
//           fuelCostSavings:
//             res.fuel_cost_savings_usd ?? 0,

//           timeSaved: res.time_saved_hours ?? 0,
//           distanceRerouted: Math.abs(
//             baselineDistance - optimizedDistance
//           ),

//           ciiRating: res.cii_rating ?? "C",

//           baselineDistance,
//           optimizedDistance,

//           baselineRoute: res.baseline_route ?? [],
//           optimizedRoute: res.optimized_route ?? [],

//           routeComparison:
//             res.route_comparison ?? {},

//           weatherSamples:
//             res.weather_samples ?? [],

//           timestamp: res.timestamp ?? "",
//         });
//       } catch (error) {
//         console.error("Optimization error:", error);
//         alert("Backend connection failed.");
//       } finally {
//         setIsOptimizing(false);
//       }
//     },
//     [setData]
//   );

//   const downloadReport = async () => {
//     if (!data) return;

//     try {
//       const response = await axios.post(
//         `${import.meta.env.VITE_API_URL}/generate-report`,
//         data,
//         { responseType: "blob" }
//       );

//       const url = window.URL.createObjectURL(
//         new Blob([response.data])
//       );

//       const link =
//         document.createElement("a");
//       link.href = url;
//       link.setAttribute(
//         "download",
//         "NaviGreen_Report.pdf"
//       );

//       document.body.appendChild(link);
//       link.click();
//       link.remove();
//     } catch (error) {
//       console.error("PDF download error:", error);
//       alert("Failed to generate report.");
//     }
//   };

//   return (
//     <div className="min-h-screen ocean-gradient flex flex-col">
//       <Navbar />

//       <main className="flex-1">
//         <HeroSection
//           onOptimize={handleOptimize}
//           isOptimizing={isOptimizing}
//         />

//         <div className="section-divider max-w-7xl mx-auto" />

//         <KPICards data={data} visible={!!data} />

//         {/* 📄 Download Button */}
//         {data && (
//           <div className="text-center mt-6">
//             <button
//               onClick={downloadReport}
//               className="bg-primary text-white px-6 py-3 rounded-xl shadow-lg hover:scale-105 transition"
//             >
//               📄 Download Optimization Report
//             </button>
//           </div>
//         )}

//         {/* 📊 Route Comparison Table */}
//         {data?.routeComparison &&
//           Object.keys(data.routeComparison)
//             .length > 0 && (
//             <div className="glass-card p-6 max-w-7xl mx-auto mt-6">
//               <h3 className="text-lg font-semibold mb-6 text-center">
//                 Route Comparison
//               </h3>

//               <div className="overflow-x-auto">
//                 <table className="w-full text-sm text-center">
//                   <thead>
//                     <tr className="border-b border-border/50">
//                       <th className="py-3">
//                         Route
//                       </th>
//                       <th className="py-3">
//                         Distance (nm)
//                       </th>
//                       <th className="py-3">
//                         Fuel (tons)
//                       </th>
//                     </tr>
//                   </thead>

//                   <tbody>
//                     {Object.entries(
//                       data.routeComparison
//                     ).map(([key, val]) => (
//                       <tr
//                         key={key}
//                         className="border-b border-border/30 hover:bg-white/5 transition"
//                       >
//                         <td className="py-3 font-medium">
//                           {key}
//                         </td>
//                         <td className="py-3">
//                           {val.distance}
//                         </td>
//                         <td className="py-3">
//                           {val.fuel}
//                         </td>
//                       </tr>
//                     ))}
//                   </tbody>
//                 </table>
//               </div>
//             </div>
//           )}

//         <MapSection
//           baselineRoute={data?.baselineRoute}
//           optimizedRoute={data?.optimizedRoute}
//           weatherSamples={data?.weatherSamples}
//           visible={!!data}
//         />

//         <AISummaryPanel
//           summary={
//             data
//               ? `Selected Route: ${data.selectedRoute} | Baseline: ${data.baselineDistance} nm | Optimized: ${data.optimizedDistance} nm`
//               : null
//           }
//           visible={!!data}
//         />
//       </main>

//       <Footer />
//     </div>
//   );
// };

// export default Index;

import { useState, useCallback } from "react";
import axios from "axios";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import HeroSection, { RouteInput } from "@/components/HeroSection";
import KPICards from "@/components/KPICards";
import MapSection from "@/components/MapSection";
import AISummaryPanel from "@/components/AISummaryPanel";
import { useOptimization } from "@/context/OptimizationContext";

const Index = () => {
  const [isOptimizing, setIsOptimizing] = useState(false);
  const { data, setData } = useOptimization();

  const handleOptimize = useCallback(
    async (input: RouteInput) => {
      try {
        setIsOptimizing(true);

        const response = await axios.post(
          `${import.meta.env.VITE_API_URL}/optimize`,
          {
            voyage: {
              start_port: {
                lat: Number(input.startLat),
                lon: Number(input.startLon),
              },
              end_port: {
                lat: Number(input.endLat),
                lon: Number(input.endLon),
              },
            },
          }
        );

        const res = response.data;

        setData({
          fuelSaved: res.fuel_reduction_percent ?? 0,
          co2Reduction: res.co2_reduction_tons ?? 0,
          timeSaved: res.time_saved_hours ?? 0,
          distanceRerouted: Math.abs(
            (res.baseline_distance_nm ?? 0) -
              (res.optimized_distance_nm ?? 0)
          ),
          fuelCostSavings: res.fuel_cost_savings_usd ?? 0,
          ciiRating: res.cii_rating ?? "C",
          baselineDistance: res.baseline_distance_nm ?? 0,
          optimizedDistance: res.optimized_distance_nm ?? 0,
          baselineRoute: res.baseline_route ?? [],
          optimizedRoute: res.optimized_route ?? [],
          routeComparison: res.route_comparison ?? {},
          weatherSamples: res.weather_samples ?? [],
          timestamp: res.timestamp ?? "",
          selectedRoute: res.selected_route ?? "Direct",
        });
      } catch (error) {
        console.error("Optimization error:", error);
        alert("Backend connection failed.");
      } finally {
        setIsOptimizing(false);
      }
    },
    [setData]
  );

  const downloadReport = async () => {
    if (!data) return;

    try {
      const formattedPayload = {
        selected_route: data.selectedRoute,
        fuel_reduction_percent: data.fuelSaved,
        co2_reduction_tons: data.co2Reduction,
        fuel_cost_savings_usd: data.fuelCostSavings,
        time_saved_hours: data.timeSaved,
        cii_rating: data.ciiRating,
        baseline_distance_nm: data.baselineDistance,
        optimized_distance_nm: data.optimizedDistance,
        route_comparison: data.routeComparison,
        timestamp: data.timestamp,
      };

      const response = await axios.post(
        `${import.meta.env.VITE_API_URL}/generate-report`,
        formattedPayload,
        { responseType: "blob" }
      );

      const url = window.URL.createObjectURL(
        new Blob([response.data])
      );

      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", "NaviGreen_Report.pdf");
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (error) {
      console.error("PDF download error:", error);
      alert("Failed to generate report.");
    }
  };

  return (
    <div className="min-h-screen ocean-gradient flex flex-col">
      <Navbar />
      <main className="flex-1">
        <HeroSection
          onOptimize={handleOptimize}
          isOptimizing={isOptimizing}
        />

        <div className="section-divider max-w-7xl mx-auto" />

        <KPICards data={data} visible={!!data} />

        {data && (
          <div className="text-center mt-6">
            <button
              onClick={downloadReport}
              className="bg-primary text-white px-6 py-3 rounded-xl shadow-lg hover:scale-105 transition"
            >
              📄 Download Optimization Report
            </button>
          </div>
        )}

        {data?.routeComparison && (
          <div className="glass-card p-6 max-w-7xl mx-auto mt-6">
            <h3 className="text-lg font-semibold mb-6 text-center">
              Route Comparison
            </h3>

            <div className="overflow-x-auto">
              <table className="w-full text-sm text-center">
                <thead>
                  <tr className="border-b border-border/50">
                    <th className="py-3">Route</th>
                    <th className="py-3">Distance (nm)</th>
                    <th className="py-3">Fuel (tons)</th>
                  </tr>
                </thead>

                <tbody>
                  {Object.entries(data.routeComparison).map(
                    ([key, val]: any) => (
                      <tr
                        key={key}
                        className="border-b border-border/30 hover:bg-white/5 transition"
                      >
                        <td className="py-3 font-medium">{key}</td>
                        <td className="py-3">{val.distance}</td>
                        <td className="py-3">{val.fuel}</td>
                      </tr>
                    )
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        <MapSection
          baselineRoute={data?.baselineRoute}
          optimizedRoute={data?.optimizedRoute}
          weatherSamples={data?.weatherSamples}
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

export default Index;