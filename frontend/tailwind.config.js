/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        base: {
          950: "#0A0E14", // page background — deep navy, not pure black
          900: "#10151D", // panel background
          800: "#161C26", // card background
          700: "#1E2733", // raised surface / hover
        },
        line: "#232C39", // hairline borders
        text: {
          primary: "#E7ECF2",
          muted: "#8B96A6",
          faint: "#5B6577",
        },
        known: {
          DEFAULT: "#33D6C0",
          dim: "#1F4A46",
        },
        novel: {
          DEFAULT: "#F5A623",
          dim: "#4A3A1C",
        },
        suspicious: {
          DEFAULT: "#E8825C",
          dim: "#43281C",
        },
        danger: {
          DEFAULT: "#E5484D",
          dim: "#451F21",
        },
      },
      fontFamily: {
        display: ["'Space Grotesk'", "sans-serif"],
        body: ["'IBM Plex Sans'", "sans-serif"],
      },
      borderRadius: {
        sm: "4px",
        DEFAULT: "6px",
        lg: "10px",
      },
      backgroundImage: {
        "grid-fade":
          "linear-gradient(180deg, rgba(51,214,192,0.06) 0%, rgba(10,14,20,0) 60%)",
      },
    },
  },
  plugins: [],
};
