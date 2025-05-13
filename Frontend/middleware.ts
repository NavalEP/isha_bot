import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export function middleware(request: NextRequest) {
  // Check for token in cookies
  const token = request.cookies.get('auth_token')?.value;
  const isLoginPage = request.nextUrl.pathname === '/login';
  
  // Get doctorId and doctor_name from query parameters if they exist
  const doctorId = request.nextUrl.searchParams.get('doctorId');
  const doctorName = request.nextUrl.searchParams.get('doctor_name');

  // Log authentication state for debugging
  console.log(`Middleware: Path=${request.nextUrl.pathname}, Token exists=${!!token}, isLoginPage=${isLoginPage}`);
  if (doctorId) console.log(`Middleware: doctorId=${doctorId}`);
  if (doctorName) console.log(`Middleware: doctorName=${doctorName}`);

  // If the user is trying to access any page other than login and is not authenticated
  if (!token && !isLoginPage) {
    console.log('Middleware: Redirecting unauthenticated user to login page');
    
    // Create login URL and only add doctor parameters if they exist
    const loginUrl = new URL('/login', request.url);
    if (doctorId) loginUrl.searchParams.set('doctorId', doctorId);
    if (doctorName) loginUrl.searchParams.set('doctor_name', doctorName);
    
    return NextResponse.redirect(loginUrl);
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