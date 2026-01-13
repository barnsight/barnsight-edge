import React, { useState, useEffect } from 'react';
import { Activity, Shield, Camera, BarChart3, AlertCircle } from 'lucide-react';

const App = () => {
  const [metrics, setMetrics] = useState({ detections: 0, fps: 0 });
  const [status, setStatus] = useState('online');

  useEffect(() => {
    // Mock metrics updates for visual effect
    const interval = setInterval(() => {
      setMetrics(prev => ({
        detections: Math.floor(Math.random() * 5),
        fps: 28 + Math.floor(Math.random() * 4)
      }));
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans selection:bg-aqua-500/30">
      {/* Background Glow */}
      <div className="fixed top-0 left-0 w-full h-full overflow-hidden pointer-events-none -z-10">
        <div className="absolute -top-1/4 -right-1/4 w-1/2 h-1/2 bg-aqua-500/10 blur-[120px] rounded-full" />
        <div className="absolute -bottom-1/4 -left-1/4 w-1/2 h-1/2 bg-blue-600/10 blur-[120px] rounded-full" />
      </div>

      {/* Navigation */}
      <nav className="border-b border-slate-800/60 bg-slate-950/50 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-aqua-500 rounded-lg flex items-center justify-center shadow-[0_0_15px_rgba(0,255,255,0.4)]">
                <Shield className="w-5 h-5 text-slate-950" />
              </div>
              <span className="text-xl font-bold tracking-tight bg-gradient-to-r from-aqua-400 to-blue-400 bg-clip-text text-transparent">
                BarnSight <span className="text-slate-500 font-medium">Edge</span>
              </span>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2 px-3 py-1 bg-emerald-500/10 border border-emerald-500/20 rounded-full">
                <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
                <span className="text-xs font-medium text-emerald-400 uppercase tracking-wider">System Live</span>
              </div>
            </div>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Main Feed */}
          <div className="lg:col-span-2 space-y-6">
            <div className="relative group rounded-3xl overflow-hidden border border-slate-800 bg-slate-900/50 shadow-2xl transition-all hover:border-aqua-500/50">
              <div className="absolute top-4 left-4 z-10 flex items-center gap-2 px-3 py-1.5 bg-slate-950/80 backdrop-blur-sm rounded-lg border border-slate-700">
                <Camera className="w-4 h-4 text-aqua-400" />
                <span className="text-xs font-bold uppercase tracking-widest text-slate-300">Live Feed - Cam 01</span>
              </div>
              
              {/* Video Placeholder - Replace with /video_feed in production */}
              <div className="aspect-video bg-slate-950 flex items-center justify-center">
                <img 
                  src="/video_feed" 
                  alt="Live Stream" 
                  className="w-full h-full object-cover"
                  onError={(e) => {
                    e.target.style.display = 'none';
                    e.target.nextSibling.style.display = 'flex';
                  }}
                />
                <div className="hidden flex-col items-center gap-4 text-slate-500">
                  <Activity className="w-12 h-12 animate-pulse" />
                  <p className="text-sm font-medium tracking-wide">Connecting to device stream...</p>
                </div>
              </div>

              {/* Overlay Controls */}
              <div className="absolute bottom-4 right-4 flex gap-2">
                <button className="p-2 bg-slate-950/80 hover:bg-aqua-500 hover:text-slate-950 rounded-xl border border-slate-700 transition-all duration-300 shadow-lg">
                  <BarChart3 className="w-5 h-5" />
                </button>
              </div>
            </div>

            {/* Quick Actions */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              {[
                { label: 'Sensitivity', value: 'High', icon: Shield },
                { label: 'Active Alerts', value: metrics.detections, icon: AlertCircle },
                { label: 'Frame Rate', value: `${metrics.fps} FPS`, icon: Activity },
                { label: 'Latency', value: '42ms', icon: Shield },
              ].map((stat, i) => (
                <div key={i} className="p-4 rounded-2xl bg-slate-900/40 border border-slate-800/60 hover:border-aqua-500/30 transition-colors group">
                  <div className="flex items-center gap-3 mb-2">
                    <stat.icon className="w-4 h-4 text-slate-500 group-hover:text-aqua-400 transition-colors" />
                    <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">{stat.label}</span>
                  </div>
                  <div className="text-lg font-bold text-slate-200">{stat.value}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Sidebar Info */}
          <div className="space-y-6">
            <div className="p-6 rounded-3xl bg-slate-900/40 border border-slate-800/60 h-full">
              <h3 className="text-sm font-bold text-slate-400 uppercase tracking-[0.2em] mb-6 flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-aqua-400" />
                Detection History
              </h3>
              
              <div className="space-y-4">
                {[...Array(5)].map((_, i) => (
                  <div key={i} className="flex items-center justify-between p-3 rounded-xl bg-slate-950/40 border border-slate-800/40">
                    <div className="flex items-center gap-3">
                      <div className={`w-2 h-2 rounded-full ${i === 0 ? 'bg-aqua-400 shadow-[0_0_8px_rgba(0,255,255,0.4)]' : 'bg-slate-700'}`} />
                      <div className="text-xs font-medium text-slate-400">12:45:0{i} PM</div>
                    </div>
                    <div className="text-xs font-bold text-slate-300">New Contaminant Detected</div>
                  </div>
                ))}
              </div>

              <div className="mt-8 pt-6 border-t border-slate-800">
                <button className="w-full py-3 bg-aqua-500/10 hover:bg-aqua-500 hover:text-slate-950 text-aqua-400 font-bold rounded-2xl border border-aqua-500/20 transition-all duration-300">
                  Generate Full Report
                </button>
              </div>
            </div>
          </div>

        </div>
      </main>

      <style jsx global>{`
        @keyframes float {
          0% { transform: translateY(0px); }
          50% { transform: translateY(-10px); }
          100% { transform: translateY(0px); }
        }
        .animate-float {
          animation: float 6s ease-in-out infinite;
        }
      `}</style>
    </div>
  );
};

export default App;
