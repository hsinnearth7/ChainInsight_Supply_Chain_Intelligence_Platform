interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  text?: string;
}

export default function LoadingSpinner({ size = 'md', text }: LoadingSpinnerProps) {
  const sizeClass = size === 'sm' ? 'w-5 h-5' : size === 'lg' ? 'w-10 h-10' : 'w-7 h-7';

  return (
    <div className="flex flex-col items-center justify-center gap-2 py-8">
      <div className={`${sizeClass} border-2 border-ci-primary/30 border-t-ci-primary rounded-full animate-spin`} />
      {text && <span className="text-sm text-ci-gray">{text}</span>}
    </div>
  );
}
