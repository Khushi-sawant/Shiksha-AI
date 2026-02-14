import React from 'react'

type ButtonProps = {
  children: React.ReactNode
  variant?: 'primary' | 'secondary' | 'outline'
  loading?: boolean
  fullWidth?: boolean
  type?: 'button' | 'submit'
}

export function Button({
  children,
  variant = 'primary',
  loading = false,
  fullWidth = false,
  type = 'button',
}: ButtonProps) {
  const styles: Record<string, React.CSSProperties> = {
    primary: {
      backgroundColor: '#2563eb',
      color: '#fff',
      border: 'none',
    },
    secondary: {
      backgroundColor: '#6b7280',
      color: '#fff',
      border: 'none',
    },
    outline: {
      backgroundColor: 'transparent',
      color: '#2563eb',
      border: '1px solid #2563eb',
    },
  }

  return (
    <button
      type={type}
      disabled={loading}
      style={{
        padding: '10px 16px',
        borderRadius: 6,
        cursor: 'pointer',
        width: fullWidth ? '100%' : 'auto',
        opacity: loading ? 0.7 : 1,
        ...styles[variant],
      }}
    >
      {loading ? 'Loading...' : children}
    </button>
  )
}
