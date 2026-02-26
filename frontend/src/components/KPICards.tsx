// import { useEffect, useRef, useState } from "react";
// import { Fuel, Leaf, Clock, Route } from "lucide-react";

// interface KPIData {
//   fuelSaved: number;
//   co2Reduction: number;
//   timeSaved: number;
//   distanceRerouted: number;
// }

// interface KPICardsProps {
//   data: KPIData | null;
//   visible: boolean;
// }

// const useCountUp = (end: number, duration: number, start: boolean) => {
//   const [value, setValue] = useState(0);
//   const rafRef = useRef<number>();

//   useEffect(() => {
//     if (!start) { setValue(0); return; }
//     const startTime = performance.now();
//     const tick = (now: number) => {
//       const elapsed = now - startTime;
//       const progress = Math.min(elapsed / duration, 1);
//       const eased = 1 - Math.pow(1 - progress, 3);
//       setValue(eased * end);
//       if (progress < 1) rafRef.current = requestAnimationFrame(tick);
//     };
//     rafRef.current = requestAnimationFrame(tick);
//     return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
//   }, [end, duration, start]);

//   return value;
// };

// const KPICard = ({
//   icon: Icon,
//   label,
//   value,
//   unit,
//   decimals = 1,
//   delay,
//   animate,
// }: {
//   icon: typeof Fuel;
//   label: string;
//   value: number;
//   unit: string;
//   decimals?: number;
//   delay: number;
//   animate: boolean;
// }) => {
//   const count = useCountUp(value, 1800, animate);

//   return (
//     <div
//       className={`glass-card-hover kpi-glow p-6 ${animate ? "" : "opacity-0"}`}
//       style={{
//         animation: animate ? `slideUp 0.5s ease-out ${delay}ms both` : "none",
//       }}
//     >
//       <div className="flex items-center gap-3 mb-4">
//         <div className="w-9 h-9 rounded-xl bg-primary/10 flex items-center justify-center">
//           <Icon className="w-4.5 h-4.5 text-primary" />
//         </div>
//         <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">{label}</span>
//       </div>
//       <div className="flex items-baseline gap-1.5">
//         {/* <span className="text-3xl font-bold text-foreground tabular-nums">
//           {animate ? count.toFixed(decimals) : "—"}
//         </span> */}
//             <span
//       className={`text-3xl font-bold tabular-nums ${
//         value > 0
//           ? "text-green-500"
//           : value < 0
//           ? "text-red-500"
//           : "text-foreground"
//       }`}
//     >
//       {animate ? count.toFixed(decimals) : "—"}
//     </span>
//         <span className="text-sm text-muted-foreground">{unit}</span>
//       </div>
//     </div>
//   );
// };

// const KPICards = ({ data, visible }: KPICardsProps) => {
//   if (!data && !visible) return null;

//   const kpis = [
//     { icon: Fuel, label: "Fuel Saved", value: data?.fuelSaved ?? 0, unit: "%", decimals: 1 },
//     { icon: Leaf, label: "CO₂ Reduction", value: data?.co2Reduction ?? 0, unit: "tons", decimals: 1 },
//     { icon: Clock, label: "Time Saved", value: data?.timeSaved ?? 0, unit: "hrs", decimals: 1 },
//     { icon: Route, label: "Distance Rerouted", value: data?.distanceRerouted ?? 0, unit: "nm", decimals: 0 },
//     { icon: Fuel, label: "Fuel Cost Saved", value: data?.fuelCostSavings ?? 0, unit: "USD", decimals: 0 },
// { icon: Leaf, label: "IMO CII Rating", value: 0, unit: data?.ciiRating ?? "-", decimals: 0 },
//   ];

//   return (
//     <section id="analytics" className="py-12 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
//       <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
//         {kpis.map((kpi, i) => (
//           <KPICard key={kpi.label} {...kpi} delay={i * 120} animate={visible} />
//         ))}
//       </div>
//     </section>
//   );
// };

// export default KPICards;



import { useEffect, useRef, useState } from "react";
import { Fuel, Leaf, Clock, Route, DollarSign, Award } from "lucide-react";

interface KPIData {
  fuelSaved: number;
  co2Reduction: number;
  timeSaved: number;
  distanceRerouted: number;
  fuelCostSavings: number;
  ciiRating: string;
}

interface KPICardsProps {
  data: KPIData | null;
  visible: boolean;
}

const useCountUp = (end: number, duration: number, start: boolean) => {
  const [value, setValue] = useState(0);
  const rafRef = useRef<number>();

  useEffect(() => {
    if (!start) {
      setValue(0);
      return;
    }

    const startTime = performance.now();

    const tick = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setValue(eased * end);

      if (progress < 1) {
        rafRef.current = requestAnimationFrame(tick);
      }
    };

    rafRef.current = requestAnimationFrame(tick);

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [end, duration, start]);

  return value;
};

const KPICardNumber = ({
  icon: Icon,
  label,
  value,
  unit,
  decimals = 1,
  delay,
  animate,
}: {
  icon: any;
  label: string;
  value: number;
  unit: string;
  decimals?: number;
  delay: number;
  animate: boolean;
}) => {
  const count = useCountUp(value, 1800, animate);

  return (
    <div
      className={`glass-card-hover kpi-glow p-6 ${
        animate ? "" : "opacity-0"
      }`}
      style={{
        animation: animate
          ? `slideUp 0.5s ease-out ${delay}ms both`
          : "none",
      }}
    >
      <div className="flex items-center gap-3 mb-4">
        <div className="w-9 h-9 rounded-xl bg-primary/10 flex items-center justify-center">
          <Icon className="w-4.5 h-4.5 text-primary" />
        </div>
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
          {label}
        </span>
      </div>

      <div className="flex items-baseline gap-1.5">
        <span className="text-3xl font-bold text-green-500 tabular-nums">
          {animate ? count.toFixed(decimals) : "—"}
        </span>
        <span className="text-sm text-muted-foreground">{unit}</span>
      </div>

      <div className="text-center mt-6">
</div>
    </div>
  );
};

const KPICardText = ({
  icon: Icon,
  label,
  value,
  delay,
  animate,
}: {
  icon: any;
  label: string;
  value: string;
  delay: number;
  animate: boolean;
}) => {
  return (
    <div
      className={`glass-card-hover kpi-glow p-6 ${
        animate ? "" : "opacity-0"
      }`}
      style={{
        animation: animate
          ? `slideUp 0.5s ease-out ${delay}ms both`
          : "none",
      }}
    >
      <div className="flex items-center gap-3 mb-4">
        <div className="w-9 h-9 rounded-xl bg-primary/10 flex items-center justify-center">
          <Icon className="w-4.5 h-4.5 text-primary" />
        </div>
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
          {label}
        </span>
      </div>

      <div className="text-3xl font-bold text-primary">
        {value}
      </div>
    </div>
  );
};

const KPICards = ({ data, visible }: KPICardsProps) => {
  if (!data || !visible) return null;

  return (
    <section
      id="analytics"
      className="py-12 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto"
    >
      <div className="grid grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
        <KPICardNumber
          icon={Fuel}
          label="Fuel Saved"
          value={data.fuelSaved}
          unit="%"
          delay={0}
          animate={visible}
        />

        <KPICardNumber
          icon={Leaf}
          label="CO₂ Reduction"
          value={data.co2Reduction}
          unit="tons"
          delay={120}
          animate={visible}
        />

        <KPICardNumber
          icon={Clock}
          label="Time Saved"
          value={data.timeSaved}
          unit="hrs"
          delay={240}
          animate={visible}
        />

        <KPICardNumber
          icon={Route}
          label="Distance Rerouted"
          value={data.distanceRerouted}
          unit="nm"
          decimals={0}
          delay={360}
          animate={visible}
        />

        <KPICardNumber
          icon={DollarSign}
          label="Fuel Cost Saved"
          value={data.fuelCostSavings}
          unit="USD"
          decimals={0}
          delay={480}
          animate={visible}
        />

        <KPICardText
          icon={Award}
          label="IMO CII Rating"
          value={data.ciiRating}
          delay={600}
          animate={visible}
        />
      </div>
    </section>
  );
};

export default KPICards;