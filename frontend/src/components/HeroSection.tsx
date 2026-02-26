// import { useState, useCallback } from "react";
// import { Navigation } from "lucide-react";

// export interface RouteInput {
//   startLat: string;
//   startLon: string;
//   endLat: string;
//   endLon: string;
// }

// interface HeroSectionProps {
//   onOptimize: (input: RouteInput) => void;
//   isOptimizing: boolean;
// }

// const HeroSection = ({ onOptimize, isOptimizing }: HeroSectionProps) => {
//   const [input, setInput] = useState<RouteInput>({
//     startLat: "",
//     startLon: "",
//     endLat: "",
//     endLon: "",
//   });

//   const handleChange = useCallback(
//     (field: keyof RouteInput) => (e: React.ChangeEvent<HTMLInputElement>) => {
//       setInput((prev) => ({ ...prev, [field]: e.target.value }));
//     },
//     []
//   );

//   const handleSubmit = (e: React.FormEvent) => {
//     e.preventDefault();
//     console.log("FORM SUBMITTED", input);
//     onOptimize(input);
//   };

//   return (
//     <section id="optimize" className="pt-28 pb-16 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
//       <div className="grid lg:grid-cols-2 gap-12 lg:gap-16 items-center">
//         {/* Left */}
//         <div className="animate-fade-in">
//           <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-border/60 bg-secondary/40 text-xs text-muted-foreground mb-6">
//             <span className="w-1.5 h-1.5 rounded-full bg-primary animate-subtle-pulse" />
//             AI Route Engine Active
//           </div>
//           <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold leading-tight tracking-tight text-foreground mb-5">
//             AI-Powered Maritime{" "}
//             <span className="text-gradient">Route Optimization</span>
//           </h1>
//           <p className="text-base sm:text-lg text-muted-foreground leading-relaxed max-w-lg">
//             Reduce fuel consumption, lower emissions, and optimize arrival time
//             using intelligent weather-aware routing.
//           </p>
//         </div>

//         {/* Right — Input Card */}
//         <div className="animate-fade-in-delay">
//           {/* <form
//             onSubmit={handleSubmit}
//             className="glass-card p-6 sm:p-8 space-y-5"
//           > */}/
//           <div className="glass-card p-6 sm:p-8 space-y-5">
//             <div className="flex items-center gap-2 mb-1">
//               <Navigation className="w-4 h-4 text-primary" />
//               <span className="text-sm font-semibold text-foreground">Route Parameters</span>
//             </div>

//             <div className="grid grid-cols-2 gap-4">
//               <div>
//                 <label className="block text-xs text-muted-foreground mb-1.5">Start Latitude</label>
//                 <input
//                   type="number"
//                   step="any"
//                   placeholder="e.g. 1.3521"
//                   className="input-field"
//                   value={input.startLat}
//                   onChange={handleChange("startLat")}
//                   required
//                 />
//               </div>
//               <div>
//                 <label className="block text-xs text-muted-foreground mb-1.5">Start Longitude</label>
//                 <input
//                   type="number"
//                   step="any"
//                   placeholder="e.g. 103.8198"
//                   className="input-field"
//                   value={input.startLon}
//                   onChange={handleChange("startLon")}
//                   required
//                 />
//               </div>
//               <div>
//                 <label className="block text-xs text-muted-foreground mb-1.5">End Latitude</label>
//                 <input
//                   type="number"
//                   step="any"
//                   placeholder="e.g. 22.3193"
//                   className="input-field"
//                   value={input.endLat}
//                   onChange={handleChange("endLat")}
//                   required
//                 />
//               </div>
//               <div>
//                 <label className="block text-xs text-muted-foreground mb-1.5">End Longitude</label>
//                 <input
//                   type="number"
//                   step="any"
//                   placeholder="e.g. 114.1694"
//                   className="input-field"
//                   value={input.endLon}
//                   onChange={handleChange("endLon")}
//                   required
//                 />
//               </div>
//             </div>
// <button
//   type="button"
//   onClick={() => {
//     console.log("BUTTON CLICKED", input);
//     onOptimize(input);
//   }}
//   className="glow-button w-full mt-2"
//   disabled={isOptimizing}
// ></button>
//             {/* <button type="submit" className="glow-button w-full mt-2" disabled={isOptimizing}>
//               {isOptimizing ? (
//                 <span className="flex items-center justify-center gap-2">
//                   <span className="w-4 h-4 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
//                   Optimizing…
//                 </span>
//               ) : (
//                 "Run Optimization"
//               )}
//             </button> */}
//           {/* </form> */}
//         </div>
//         </div>
//       </div>
//     </section>
//   );
// };

// export default HeroSection;
import { useState, useCallback } from "react";
import { Navigation } from "lucide-react";

export interface RouteInput {
  startLat: string;
  startLon: string;
  endLat: string;
  endLon: string;
}

interface HeroSectionProps {
  onOptimize: (input: RouteInput) => void;
  isOptimizing: boolean;
}

const HeroSection = ({ onOptimize, isOptimizing }: HeroSectionProps) => {
  const [input, setInput] = useState<RouteInput>({
    startLat: "",
    startLon: "",
    endLat: "",
    endLon: "",
  });

  const handleChange = useCallback(
    (field: keyof RouteInput) => (e: React.ChangeEvent<HTMLInputElement>) => {
      setInput((prev) => ({ ...prev, [field]: e.target.value }));
    },
    []
  );

  return (
    <section
      id="optimize"
      className="pt-28 pb-16 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto"
    >
      <div className="grid lg:grid-cols-2 gap-12 lg:gap-16 items-center">
        {/* Left */}
        <div className="animate-fade-in">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-border/60 bg-secondary/40 text-xs text-muted-foreground mb-6">
            <span className="w-1.5 h-1.5 rounded-full bg-primary animate-subtle-pulse" />
            AI Route Engine Active
          </div>

          <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold leading-tight tracking-tight text-foreground mb-5">
            AI-Powered Maritime{" "}
            <span className="text-gradient">Route Optimization</span>
          </h1>

          <p className="text-base sm:text-lg text-muted-foreground leading-relaxed max-w-lg">
            Reduce fuel consumption, lower emissions, and optimize arrival time
            using intelligent weather-aware routing.
          </p>
        </div>

        {/* Right — Input Card */}
        <div className="animate-fade-in-delay">
          <div className="glass-card p-6 sm:p-8 space-y-5">
            <div className="flex items-center gap-2 mb-1">
              <Navigation className="w-4 h-4 text-primary" />
              <span className="text-sm font-semibold text-foreground">
                Route Parameters
              </span>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-muted-foreground mb-1.5">
                  Start Latitude
                </label>
                <input
                  type="number"
                  step="any"
                  className="input-field"
                  value={input.startLat}
                  onChange={handleChange("startLat")}
                  required
                />
              </div>

              <div>
                <label className="block text-xs text-muted-foreground mb-1.5">
                  Start Longitude
                </label>
                <input
                  type="number"
                  step="any"
                  className="input-field"
                  value={input.startLon}
                  onChange={handleChange("startLon")}
                  required
                />
              </div>

              <div>
                <label className="block text-xs text-muted-foreground mb-1.5">
                  End Latitude
                </label>
                <input
                  type="number"
                  step="any"
                  className="input-field"
                  value={input.endLat}
                  onChange={handleChange("endLat")}
                  required
                />
              </div>

              <div>
                <label className="block text-xs text-muted-foreground mb-1.5">
                  End Longitude
                </label>
                <input
                  type="number"
                  step="any"
                  className="input-field"
                  value={input.endLon}
                  onChange={handleChange("endLon")}
                  required
                />
              </div>
            </div>

            <button
              type="button"
              onClick={() => {
                console.log("BUTTON CLICKED", input);
                onOptimize(input);
              }}
              className="glow-button w-full mt-2"
              disabled={isOptimizing}
            >
              {isOptimizing ? "Optimizing..." : "Run Optimization"}
            </button>
          </div>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;