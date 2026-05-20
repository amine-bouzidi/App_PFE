"use client";

import { useSession, signOut } from "next-auth/react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { 
  LayoutDashboard, 
  Search, 
  FolderKanban, 
  LineChart, 
  FileText, 
  Settings, 
  LogOut,
  TrendingUp,
  User,
  BrainCircuit
} from "lucide-react";
import { cn } from "@/lib/utils";

const links = [
  { href: "/dashboard", label: "Tableau de bord", icon: LayoutDashboard },
  { href: "/scraper", label: "Module de Scraping", icon: Search },
  { href: "/projects", label: "Projets", icon: FolderKanban },
  { href: "/analyses", label: "Analyses IA", icon: LineChart },
  { href: "/advanced-analysis", label: "Analyse avancee", icon: BrainCircuit },
  { href: "/reports", label: "Rapports", icon: FileText },
  { href: "/settings", label: "Paramètres", icon: Settings },
];

export default function Sidebar() {
  const { data: session } = useSession();
  const pathname = usePathname();

  return (
    <aside className="fixed inset-y-0 left-0 z-20 flex w-64 flex-col border-r border-slate-900 bg-slate-950/80 backdrop-blur-md">
      {/* Top Brand Logo */}
      <div className="flex h-16 items-center border-b border-slate-900 px-6">
        <Link href="/dashboard" className="flex items-center gap-2.5">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-tr from-violet-600 to-fuchsia-600 text-white font-black shadow-lg shadow-violet-500/20">
            IF
          </div>
          <div>
            <span className="font-bold text-white leading-none">InsightFlow</span>
            <span className="block text-[10px] text-violet-400 font-medium uppercase tracking-wider">Veille Sociale par IA</span>
          </div>
        </Link>
      </div>

      {/* Navigation items */}
      <nav className="flex-1 space-y-1.5 px-4 py-6">
        {links.map((link) => {
          const Icon = link.icon;
          const isActive = pathname === link.href;

          return (
            <Link
              key={link.href}
              href={link.href}
              className={cn(
                "flex items-center gap-3.5 rounded-lg px-3 py-2.5 text-sm font-medium transition active:scale-[0.98]",
                isActive
                  ? "bg-violet-600 text-white shadow-lg shadow-violet-600/10"
                  : "text-slate-400 hover:bg-slate-900/60 hover:text-white"
              )}
            >
              <Icon className={cn("h-4.5 w-4.5", isActive ? "text-white" : "text-slate-400")} />
              {link.label}
            </Link>
          );
        })}
      </nav>

      {/* User profile block */}
      <div className="mt-auto border-t border-slate-900 p-4">
        <div className="flex items-center gap-3 rounded-lg bg-slate-900/40 p-3 border border-slate-900">
          <div className="flex h-9 w-9 items-center justify-center rounded-full bg-violet-500/10 text-violet-400">
            <User className="h-4.5 w-4.5" />
          </div>
          <div className="flex-1 overflow-hidden">
            <span className="block truncate text-sm font-semibold text-white">
              {session?.user?.name || "Utilisateur Premium"}
            </span>
            <span className="block truncate text-xs text-slate-400">
              {session?.user?.email || "utilisateur@insightflow.ai"}
            </span>
          </div>
        </div>

        <button
          onClick={() => signOut({ callbackUrl: "/login" })}
          className="mt-3 flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium text-red-400 transition hover:bg-red-500/10 hover:text-red-300"
        >
          <LogOut className="h-4 w-4" />
          Se déconnecter
        </button>
      </div>
    </aside>
  );
}
