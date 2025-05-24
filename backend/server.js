const express = require("express");
const mysql = require("mysql2");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const cors = require("cors");
const dotenv = require("dotenv");
const passport = require("passport");
const GoogleStrategy = require("passport-google-oauth20").Strategy;
const session = require("express-session");
const axios = require("axios");
const cheerio = require("cheerio");
const puppeteer = require("puppeteer");


// Initialize environment variables
dotenv.config();

const app = express();
app.use(express.json());
app.use(cors());

// MySQL connection
const db = mysql.createConnection({
  host: "localhost",
  user: "root",
  password: "",
  database: "user_auth",
});

db.connect((err) => {
  if (err) throw err;
  console.log(" Connected to the database");
});

// JWT secret key
const JWT_SECRET = process.env.JWT_SECRET || "your-secret-key";

// Initialize Passport for Google Authentication
passport.use(
  new GoogleStrategy(
    {
      clientID: process.env.GOOGLE_CLIENT_ID,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET,
      callbackURL: "http://localhost:3000/auth/google/callback",
    },
    (accessToken, refreshToken, profile, done) => {
      const email = profile.emails[0].value;
      db.query("SELECT * FROM users WHERE email = ?", [email], (err, result) => {
        if (err) return done(err);
        if (result.length > 0) return done(null, result[0]);
        db.query(
          "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
          [profile.displayName, email, null],
          (err, result) => {
            if (err) return done(err);
            return done(null, { id: result.insertId, name: profile.displayName, email });
          }
        );
      });
    }
  )
);

app.use(passport.initialize());
app.use(
  session({
    secret: process.env.SESSION_SECRET || "defaultSecret",
    resave: false,
    saveUninitialized: true,
  })
);

// Google Authentication Routes
app.get("/auth/google", passport.authenticate("google", { scope: ["profile", "email"] }));

app.get("/auth/google/callback", passport.authenticate("google", { session: false }), (req, res) => {
  const token = jwt.sign({ id: req.user.id, email: req.user.email }, JWT_SECRET, { expiresIn: "1h" });
  res.redirect(`http://localhost:3001/?token=${token}`);
});

// Sign Up Endpoint
app.post("/api/signup", async (req, res) => {
  const { name, email, password } = req.body;
  db.query("SELECT * FROM users WHERE email = ?", [email], async (err, result) => {
    if (err) return res.status(500).json({ message: "Database error" });
    if (result.length > 0) return res.status(400).json({ message: "User already exists" });
    const hashedPassword = await bcrypt.hash(password, 10);
    db.query("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", [name, email, hashedPassword], (err) => {
      if (err) return res.status(500).json({ message: "Database error" });
      res.status(201).json({ message: "User registered successfully" });
    });
  });
});

// Login Endpoint
app.post("/api/login", (req, res) => {
  const { email, password } = req.body;
  db.query("SELECT * FROM users WHERE email = ?", [email], async (err, result) => {
    if (err) return res.status(500).json({ message: "Database error" });
    if (result.length === 0) return res.status(400).json({ message: "User not found" });
    const user = result[0];
    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) return res.status(400).json({ message: "Invalid password" });
    const token = jwt.sign({ id: user.id, email: user.email }, JWT_SECRET, { expiresIn: "1h" });
    res.status(200).json({ message: "Login successful", token });
  });
});

// Password Reset Endpoint
app.post("/api/reset-password", (req, res) => {
  const { email } = req.body;
  db.query("SELECT * FROM users WHERE email = ?", [email], (err, result) => {
    if (err) return res.status(500).json({ message: "Database error" });
    if (result.length === 0) return res.status(400).json({ message: "User not found" });
    res.status(200).json({ message: "Password reset link sent to your email" });
  });
});

// Web Scraping Endpoint to Get Shein Product Image
app.post("/get-product-image", async (req, res) => {
  const { url } = req.body;
  if (!url) return res.status(400).json({ error: "Please provide a product URL" });
  try {
    const headers = {
      "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    };
    const { data } = await axios.get(url, { headers });
    const $ = cheerio.load(data);
    const imgElement = $("img[src*='asos-media']");
const imgUrl = imgElement.attr("src");

    if (imgUrl) {
      res.json({ imageUrl: imgUrl });
    } else {
      res.status(404).json({ error: "Product image not found" });
    }
  } catch (error) {
    console.error(" Error fetching product image:", error);
    res.status(500).json({ error: "Error processing request" });
  }
});

app.post("/get-product-image", async (req, res) => {
  const { url } = req.body;
  if (!url) return res.status(400).json({ error: "Please provide a product URL" });

  try {
      const browser = await puppeteer.launch({ headless: true });
      const page = await browser.newPage();

      await page.goto(url, { waitUntil: "networkidle2" });

      let imgUrl = "";

      if (url.includes("shein.com")) {
          // البحث عن صورة بالحجم الكامل
          imgUrl = await page.$eval("img.j-image", (img) => img.getAttribute("src"));
      } else if (url.includes("asos.com")) {
          // انتظار الصور الكبيرة وتحميل الصورة بالحجم الكامل
          await page.waitForSelector("img.gallery-image");
          const images = await page.$$eval("img.gallery-image", (imgs) =>
              imgs.map((img) => img.getAttribute("data-src") || img.getAttribute("src"))
          );
          imgUrl = images.length > 0 ? images[0] : "";
      }

      await browser.close();

      if (imgUrl) {
          res.json({ imageUrl: imgUrl });
      } else {
          res.status(404).json({ error: "Product image not found" });
      }
  } catch (error) {
      console.error("Error fetching product image:", error);
      res.status(500).json({ error: "Error processing request" });
  }
});

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(` Server running on port ${PORT}`));
