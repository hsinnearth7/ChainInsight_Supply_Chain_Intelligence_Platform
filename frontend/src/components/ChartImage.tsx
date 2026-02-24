import { useState } from 'react';
import LoadingSpinner from './LoadingSpinner';

interface ChartImageProps {
  src: string;
  alt: string;
  className?: string;
}

export default function ChartImage({ src, alt, className = '' }: ChartImageProps) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  return (
    <div className={`relative bg-white dark:bg-ci-dark-card rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden ${className}`}>
      {loading && !error && (
        <div className="absolute inset-0 flex items-center justify-center">
          <LoadingSpinner size="sm" />
        </div>
      )}
      {error ? (
        <div className="flex items-center justify-center h-48 text-ci-gray text-sm">
          Chart not available
        </div>
      ) : (
        <img
          src={src}
          alt={alt}
          className={`w-full h-auto transition-opacity ${loading ? 'opacity-0' : 'opacity-100'}`}
          onLoad={() => setLoading(false)}
          onError={() => { setError(true); setLoading(false); }}
          loading="lazy"
        />
      )}
    </div>
  );
}
