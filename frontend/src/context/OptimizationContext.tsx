// // import { createContext, useContext, useState, useEffect } from "react";

// // interface OptimizationData {
// //   fuelSaved: number;
// //   co2Reduction: number;
// //   timeSaved: number;
// //   distanceRerouted: number;
// //   baselineDistance: number;
// //   optimizedDistance: number;
// //   start: { lat: number; lon: number };
// //   end: { lat: number; lon: number };
// // }

// // interface OptimizationContextType {
// //   data: OptimizationData | null;
// //   setData: (data: OptimizationData) => void;
// // }

// // const OptimizationContext = createContext<OptimizationContextType | undefined>(undefined);

// // export const OptimizationProvider = ({ children }: { children: React.ReactNode }) => {
// //   const [data, setDataState] = useState<OptimizationData | null>(null);

// //   useEffect(() => {
// //     const stored = localStorage.getItem("optimizationData");
// //     if (stored) {
// //       setDataState(JSON.parse(stored));
// //     }
// //   }, []);

// //   const setData = (newData: OptimizationData) => {
// //     localStorage.setItem("optimizationData", JSON.stringify(newData));
// //     setDataState(newData);
// //   };

// //   return (
// //     <OptimizationContext.Provider value={{ data, setData }}>
// //       {children}
// //     </OptimizationContext.Provider>
// //   );
// // };

// // export const useOptimization = () => {
// //   const context = useContext(OptimizationContext);
// //   if (!context) {
// //     throw new Error("useOptimization must be used within OptimizationProvider");
// //   }
// //   return context;
// // // };
// import { createContext, useContext, useState, useEffect } from "react";

// interface OptimizationData {
//   fuelSaved: number;
//   co2Reduction: number;
//   fuelCostSavings: number;
//   ciiRating: string;
//   baselineDistance: number;
//   optimizedDistance: number;
//   baselineRoute: any[];
//   optimizedRoute: any[];
//   routeComparison: any;
//   timestamp: string;
// }
// interface OptimizationContextType {
//   data: OptimizationData | null;
//   setData: (data: OptimizationData) => void;
// }

// const OptimizationContext = createContext<OptimizationContextType | null>(null);

// export const OptimizationProvider = ({ children }: { children: React.ReactNode }) => {
//   const [data, setDataState] = useState<OptimizationData | null>(null);

//   useEffect(() => {
//     try {
//       const stored = localStorage.getItem("optimizationData");
//       if (stored) {
//         setDataState(JSON.parse(stored));
//       }
//     } catch (error) {
//       console.error("Failed to parse stored optimization data:", error);
//       localStorage.removeItem("optimizationData");
//     }
//   }, []);

//   const setData = (newData: OptimizationData) => {
//     try {
//       localStorage.setItem("optimizationData", JSON.stringify(newData));
//       setDataState(newData);
//     } catch (error) {
//       console.error("Failed to save optimization data:", error);
//     }
//   };

//   return (
//     <OptimizationContext.Provider value={{ data, setData }}>
//       {children}
//     </OptimizationContext.Provider>
//   );
// };

// export const useOptimization = () => {
//   const context = useContext(OptimizationContext);

//   // SAFE fallback instead of crashing the whole app
//   if (!context) {
//     return {
//       data: null,
//       setData: () => {},
//     };
//   }

//   return context;
// };

// import { createContext, useContext, useState } from "react";

// export interface OptimizationData {
//   fuelSaved: number;
//   co2Reduction: number;
//   fuelCostSavings: number;
//   ciiRating: string;
//   baselineDistance: number;
//   optimizedDistance: number;
//   baselineRoute: any[];
//   optimizedRoute: any[];
//   routeComparison: any;
//   timestamp: string;
// }

// const OptimizationContext = createContext<any>(null);

// export const OptimizationProvider = ({ children }: any) => {
//   const [data, setData] = useState<OptimizationData | null>(null);

//   return (
//     <OptimizationContext.Provider value={{ data, setData }}>
//       {children}
//     </OptimizationContext.Provider>
//   );
// };

// export const useOptimization = () => useContext(OptimizationContext);



// import { createContext, useContext, useState } from "react";

// export interface WeatherPoint {
//   lat: number;
//   lon: number;
//   wind: number;
// }

// export interface OptimizationData {
//   fuelSaved: number;
//   co2Reduction: number;
//   fuelCostSavings: number;
//   ciiRating: string;

//   baselineDistance: number;
//   optimizedDistance: number;

//   baselineRoute: any[];
//   optimizedRoute: any[];

//   routeComparison: any;

//   timeSaved: number;
//   distanceRerouted: number;

//   weatherSamples: WeatherPoint[];

//   timestamp: string;
// }

// interface OptimizationContextType {
//   data: OptimizationData | null;
//   setData: (data: OptimizationData) => void;
// }

// const OptimizationContext = createContext<OptimizationContextType | null>(null);

// export const OptimizationProvider = ({ children }: { children: React.ReactNode }) => {
//   const [data, setData] = useState<OptimizationData | null>(null);

//   return (
//     <OptimizationContext.Provider value={{ data, setData }}>
//       {children}
//     </OptimizationContext.Provider>
//   );
// };

// export const useOptimization = () => {
//   const context = useContext(OptimizationContext);

//   if (!context) {
//     throw new Error("useOptimization must be used inside OptimizationProvider");
//   }

//   return context;
// };
import { createContext, useContext, useState } from "react";

export interface WeatherPoint {
  lat: number;
  lon: number;
  wind: number;
}

export interface OptimizationData {
  selectedRoute: string;

  fuelSaved: number;
  co2Reduction: number;
  fuelCostSavings: number;
  ciiRating: string;

  baselineDistance: number;
  optimizedDistance: number;

  baselineRoute: any[];
  optimizedRoute: any[];

  routeComparison: Record<
    string,
    {
      distance: number;
      fuel: number;
    }
  >;

  timeSaved: number;
  distanceRerouted: number;

  weatherSamples: WeatherPoint[];

  timestamp: string;
}

interface OptimizationContextType {
  data: OptimizationData | null;
  setData: (data: OptimizationData) => void;
}

const OptimizationContext =
  createContext<OptimizationContextType | null>(null);

export const OptimizationProvider = ({
  children,
}: {
  children: React.ReactNode;
}) => {
  const [data, setData] = useState<OptimizationData | null>(
    null
  );

  return (
    <OptimizationContext.Provider value={{ data, setData }}>
      {children}
    </OptimizationContext.Provider>
  );
};

export const useOptimization = () => {
  const context = useContext(OptimizationContext);

  if (!context) {
    throw new Error(
      "useOptimization must be used inside OptimizationProvider"
    );
  }

  return context;
};