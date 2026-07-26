import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PressLens — Football video intelligence",
  description: "Search, compare and explain build-up and pressing patterns.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
