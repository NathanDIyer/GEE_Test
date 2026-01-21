/**
 * Simple password authentication endpoint.
 * Validates password against ACCESS_PASSWORD environment variable.
 */

export default function handler(req, res) {
  // Only allow POST
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  const { password } = req.body

  if (!password) {
    return res.status(400).json({ error: 'Password required' })
  }

  const accessPassword = process.env.ACCESS_PASSWORD

  if (!accessPassword) {
    console.error('ACCESS_PASSWORD environment variable not set')
    return res.status(500).json({ error: 'Server configuration error' })
  }

  if (password === accessPassword) {
    return res.status(200).json({ success: true })
  } else {
    return res.status(401).json({ error: 'Invalid password' })
  }
}
