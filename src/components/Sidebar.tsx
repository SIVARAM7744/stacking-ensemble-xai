import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  AlertTriangle,
  Brain,
  History,
  Server,
  X,
} from 'lucide-react';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

interface NavItem {
  to: string;
  label: string;
  icon: React.ReactNode;
}

const menuItems: NavItem[] = [
  { to: '/',               label: 'Dashboard',          icon: <LayoutDashboard size={18} /> },
  { to: '/predict',        label: 'Predict Disaster',   icon: <AlertTriangle size={18} /> },
  { to: '/explainability', label: 'Explainability',     icon: <Brain size={18} /> },
  { to: '/history',        label: 'Historical Records', icon: <History size={18} /> },
  { to: '/api-status',     label: 'API Status',         icon: <Server size={18} /> },
];

export function Sidebar({ isOpen, onClose }: SidebarProps) {
  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/20 z-40 lg:hidden"
          onClick={onClose}
        />
      )}

      <aside
        className={`
          fixed top-16 left-0 h-[calc(100vh-4rem)] w-64 z-50
          transform transition-transform duration-200 lg:transform-none
          ${isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        `}
        style={{ backgroundColor: '#F3F4F6', borderRight: '1px solid #E5E7EB' }}
      >
        {/* Mobile close button */}
        <div className="flex justify-end p-2 lg:hidden">
          <button onClick={onClose}>
            <X size={20} style={{ color: '#111827' }} />
          </button>
        </div>

        <nav className="p-4 space-y-1">
          {menuItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}          /* exact match for root only */
              onClick={onClose}
              className="w-full flex items-center gap-3 px-3 py-2.5 text-sm text-left"
              style={({ isActive }) => ({
                backgroundColor: isActive ? '#1E3A8A' : 'transparent',
                color:           isActive ? '#FFFFFF' : '#111827',
                borderRadius: '4px',
                textDecoration: 'none',
                display: 'flex',
              })}
            >
              {item.icon}
              <span>{item.label}</span>
            </NavLink>
          ))}
        </nav>
      </aside>
    </>
  );
}
