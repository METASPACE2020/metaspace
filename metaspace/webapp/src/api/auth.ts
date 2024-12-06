const fetchPostJson = async (url: string, body: object) =>
  fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify(body),
  })

export const signOut = async () => {
  const response = await fetch('/api_auth/signout', { method: 'POST', credentials: 'include' })
  if (response.status !== 200) {
    throw new Error(`Unexpected response from server: ${response.status} ${response.statusText}`)
  }
}

export const signInByEmail = async (email: string, password: string): Promise<boolean> => {
  const response = await fetchPostJson('/api_auth/signin', { email, password })
  return response.status >= 200 && response.status < 300
}

export const verifyRecaptcha = async (recaptchaToken: string): Promise<boolean> => {
  const response = await fetchPostJson('/api_auth/verify_captcha', { recaptchaToken })
  return response.status >= 200 && response.status < 300
}

export const createAccountByEmail = async (email: string, password: string, name: string, recaptchaToken?: string) => {
  const response = await fetchPostJson('/api_auth/createaccount', {
    email,
    password,
    name,
    recaptchaToken,
  })
  if (response.status !== 200) {
    let error = `Unexpected response from server: ${response.status} ${response.statusText}`
    try {
      const data = await response.json()
      error = JSON.parse(data.message).message
    } catch (error) {
      // ignore
    }

    throw new Error(error)
  }
}

export const sendPasswordResetToken = async (email: string) => {
  const response = await fetchPostJson('/api_auth/sendpasswordresettoken', { email })
  if (response.status !== 200) {
    throw new Error(`Unexpected response from server: ${response.status} ${response.statusText}`)
  }
}

export const validatePasswordResetToken = async (token: string, email: string): Promise<boolean> => {
  const response = await fetchPostJson('/api_auth/validatepasswordresettoken', { token, email })
  return response.status === 200
}

export const resetPassword = async (token: string, email: string, password: string) => {
  const response = await fetchPostJson('/api_auth/resetpassword', { token, email, password })
  if (response.status !== 200) {
    throw new Error(`Unexpected response from server: ${response.status} ${response.statusText}`)
  }
}
