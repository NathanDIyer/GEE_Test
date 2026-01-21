import { useState, useEffect } from 'react'

const AUTH_KEY = 'hourly_match_authenticated'

export function useAuth() {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    // Check localStorage for existing session
    const stored = localStorage.getItem(AUTH_KEY)
    setIsAuthenticated(stored === 'true')
    setIsLoading(false)
  }, [])

  const login = async (password) => {
    // Demo password for static hosting (GitHub Pages) where API isn't available
    const DEMO_PASSWORD = 'demo123'

    try {
      const response = await fetch('/api/auth', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password }),
      })

      // API not found (404) - use demo password fallback
      if (response.status === 404) {
        if (password === DEMO_PASSWORD) {
          localStorage.setItem(AUTH_KEY, 'true')
          setIsAuthenticated(true)
          return { success: true }
        }
        return { success: false, error: 'Invalid password' }
      }

      if (response.ok) {
        localStorage.setItem(AUTH_KEY, 'true')
        setIsAuthenticated(true)
        return { success: true }
      } else {
        return { success: false, error: 'Invalid password' }
      }
    } catch (err) {
      // Network error - use demo password fallback
      if (password === DEMO_PASSWORD) {
        localStorage.setItem(AUTH_KEY, 'true')
        setIsAuthenticated(true)
        return { success: true }
      }
      return { success: false, error: 'Connection error' }
    }
  }

  const logout = () => {
    localStorage.removeItem(AUTH_KEY)
    setIsAuthenticated(false)
  }

  return { isAuthenticated, isLoading, login, logout }
}
