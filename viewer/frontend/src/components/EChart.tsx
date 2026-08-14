import { useEffect, useRef } from "react";
import * as echarts from "echarts";

type EChartProps = {
  option: unknown;
  className?: string;
  onClick?: (params: unknown) => void;
};

export function EChart({ option, className, onClick }: EChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<echarts.EChartsType | null>(null);

  useEffect(() => {
    if (!containerRef.current) {
      return;
    }
    if (!chartRef.current) {
      chartRef.current = echarts.init(containerRef.current, undefined, { renderer: "canvas" });
    }
    const chart = chartRef.current;
    const resizeObserver = new ResizeObserver(() => chart.resize());
    resizeObserver.observe(containerRef.current);
    return () => {
      resizeObserver.disconnect();
    };
  }, []);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) {
      return;
    }
    chart.setOption(option as echarts.EChartsOption, { notMerge: true, lazyUpdate: true });
  }, [option]);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) {
      return;
    }
    if (onClick) {
      chart.on("click", onClick as (params: unknown) => void);
    }
    return () => {
      if (onClick) {
        chart.off("click", onClick as (params: unknown) => void);
      }
    };
  }, [onClick]);

  useEffect(() => {
    return () => {
      const chart = chartRef.current;
      if (!chart) {
        return;
      }
      chart.dispose();
      chartRef.current = null;
    };
  }, []);

  return <div ref={containerRef} className={className ?? "chart"} />;
}
