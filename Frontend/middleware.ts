import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export function middleware(request: NextRequest) {
  // Check for token in cookies
  const token = request.cookies.get('auth_token')?.value;
  const isLoginPage = request.nextUrl.pathname === '/login';

  // Log authentication state for debugging
  console.log(`Middleware: Path=${request.nextUrl.pathname}, Token exists=${!!token}, isLoginPage=${isLoginPage}`);

  // If the user is trying to access any page other than login and is not authenticated
  if (!token && !isLoginPage) {
    console.log('Middleware: Redirecting unauthenticated user to login page');
    return NextResponse.redirect(new URL('/login', request.url));
  }

  // If the user is already authenticated and trying to access the login page
  if (token && isLoginPage) {
    console.log('Middleware: Redirecting authenticated user to home page');
    return NextResponse.redirect(new URL('/', request.url));
  }

  return NextResponse.next();
}

export const config = {
  matcher: ['/', '/login'],
}; 