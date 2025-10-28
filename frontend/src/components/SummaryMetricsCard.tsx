import React from "react";
import { Badge } from "./ui/badge";
import type { TodayStats } from "../lib/types";


interface SummaryMetricsCardProps {
  stats?: TodayStats;
  className?: string;
}


const SummaryMetricsCard: React.FC<SummaryMetricsCardProps> = ({ stats, className = "" }) => {
  return (
    <div className={`flex flex-col gap-4 bg-white/90 rounded-2xl shadow-xl p-6 ${className} animate-fade-in`}>
      <h2 className="text-lg font-bold text-blue-800 mb-2 tracking-tight">Session Metrics</h2>
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <span className="text-gray-500">Focus Score</span>
          <span className="font-bold text-blue-700 text-lg animate-fade-in">{stats?.focus_score ?? "-"}</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-gray-500">Focused Time (min)</span>
          <span className="font-bold text-green-700 text-lg animate-fade-in">{stats?.focused_time ? stats.focused_time.toFixed(1) : "-"}</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-gray-500">Distracted Time (min)</span>
          <span className="font-bold text-red-600 text-lg animate-fade-in">{stats?.distracted_time ? stats.distracted_time.toFixed(1) : "-"}</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-gray-500">Sessions</span>
          <span className="font-bold text-purple-700 animate-fade-in">{stats?.sessions ?? "-"}</span>
        </div>
      </div>
    </div>
  );
};

export default SummaryMetricsCard;
