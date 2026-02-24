interface KPICardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  color?: string;
  icon?: string;
}

export default function KPICard({ title, value, subtitle, color = 'ci-primary', icon }: KPICardProps) {
  return (
    <div className="bg-white dark:bg-ci-dark-card rounded-lg border border-gray-200 dark:border-gray-700 p-4 flex flex-col">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium text-ci-gray uppercase tracking-wide">{title}</span>
        {icon && <span className="text-lg">{icon}</span>}
      </div>
      <span className={`text-2xl font-bold text-${color}`}>{value}</span>
      {subtitle && <span className="text-xs text-ci-gray mt-1">{subtitle}</span>}
    </div>
  );
}
