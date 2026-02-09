import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Navbar from "@/components/Navbar";
import { Providers } from "@/components/Providers";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "EDGE AI - Sports Betting Analyst",
  description:
    "AI-powered sports betting analysis with expected value calculations, model breakdowns, and Kelly criterion sizing.",
  viewport: "width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no",
  themeColor: "#7c3aed",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="manifest" href="/manifest.json" />
      </head>
      <body className={`${inter.variable} font-sans antialiased`}>
        <Providers>
          <div className="min-h-screen dot-grid">
            <Navbar />
            <main className="min-h-[calc(100vh-3.5rem)]">{children}</main>
          </div>
        </Providers>
      </body>
    </html>
  );
}
