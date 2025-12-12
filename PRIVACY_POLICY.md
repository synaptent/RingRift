# RingRift Privacy Policy

**Last Updated:** 2025-12-11
**Effective Date:** 2025-12-11

> **Note:** This is a placeholder document for an open-source, limited-scale project.
> If RingRift grows significantly or targets EU users, professional legal review for
> GDPR compliance is recommended.

---

## 1. Introduction

This Privacy Policy explains how RingRift ("we", "us", or "the Service") collects, uses, and protects your personal information when you use our multiplayer strategy game.

We are committed to protecting your privacy and being transparent about our data practices.

---

## 2. Information We Collect

### 2.1 Information You Provide

| Data Type         | Purpose                                                   | Required |
| ----------------- | --------------------------------------------------------- | -------- |
| **Email address** | Account creation, password reset, important notifications | Yes      |
| **Username**      | Display name in games and leaderboards                    | Yes      |
| **Password**      | Account security (stored as secure hash, never plaintext) | Yes      |

### 2.2 Information Collected Automatically

| Data Type               | Purpose                                   | Retention               |
| ----------------------- | ----------------------------------------- | ----------------------- |
| **Game history**        | Statistics, replays, matchmaking          | Until account deletion  |
| **IP address**          | Security, rate limiting, abuse prevention | 30 days in logs         |
| **Browser/device info** | Compatibility, debugging                  | 30 days in logs         |
| **Game actions**        | Replays, AI training (anonymized)         | Indefinite (anonymized) |

### 2.3 Information We Do NOT Collect

- Payment or financial information (the Service is free)
- Precise geolocation
- Contacts or address book
- Biometric data
- Data from third-party social media accounts

---

## 3. How We Use Your Information

We use collected information to:

1. **Provide the Service**
   - Create and manage your account
   - Enable multiplayer games
   - Display leaderboards and statistics

2. **Improve the Service**
   - Analyze gameplay patterns (anonymized)
   - Debug technical issues
   - Train AI opponents (using anonymized game data)

3. **Ensure Security**
   - Detect and prevent abuse, cheating, or unauthorized access
   - Enforce our Terms of Service

4. **Communicate with You**
   - Password reset emails
   - Important service announcements
   - (Optional) Game notifications if you opt in

---

## 4. Data Sharing

### 4.1 We Do NOT Sell Your Data

We do not sell, rent, or trade your personal information to third parties.

### 4.2 Limited Sharing

We may share data only in these circumstances:

| Recipient             | Data Shared               | Purpose                   |
| --------------------- | ------------------------- | ------------------------- |
| **Other players**     | Username, game statistics | Multiplayer functionality |
| **Service providers** | As needed for hosting     | Infrastructure operation  |
| **Legal authorities** | As required by law        | Legal compliance          |

### 4.3 Anonymized Data

We may share anonymized, aggregated data (e.g., "X games played this month") for research or to improve AI. This data cannot identify individual users.

---

## 5. Data Storage and Security

### 5.1 Where Data is Stored

Your data is stored on servers located in [Your Server Location/Region].

### 5.2 Security Measures

We implement industry-standard security measures:

- **Passwords:** Hashed using bcrypt with salt (never stored in plaintext)
- **Transport:** All data transmitted over HTTPS/TLS encryption
- **Access:** Limited to authorized personnel only
- **Sessions:** JWT tokens with expiration and refresh rotation

### 5.3 Security Limitations

No system is 100% secure. While we take reasonable precautions, we cannot guarantee absolute security. Please use a strong, unique password for your account.

---

## 6. Data Retention

| Data Type              | Retention Period              |
| ---------------------- | ----------------------------- |
| Account information    | Until you delete your account |
| Game history           | Until you delete your account |
| Server logs (IP, etc.) | 30 days                       |
| Anonymized game data   | Indefinite                    |

After account deletion, your personal data is removed within 30 days. Some anonymized data may be retained for AI training and statistics.

---

## 7. Your Rights and Choices

### 7.1 Access Your Data

You can view your account information and game history through the Settings page.

### 7.2 Export Your Data

You can request a copy of your data by:

- Using the "Export My Data" feature in Settings, OR
- Contacting us at [your-email@example.com]

We will provide your data in a machine-readable format (JSON) within 30 days.

### 7.3 Delete Your Data

You can delete your account and associated data by:

- Using the "Delete Account" feature in Settings, OR
- Contacting us at [your-email@example.com]

Account deletion is permanent and cannot be undone.

### 7.4 Correct Your Data

You can update your email address and username through the Settings page.

### 7.5 Opt-Out of Communications

You can opt out of non-essential emails through the Settings page. We will still send essential emails (password reset, critical security notices).

---

## 8. Cookies and Similar Technologies

### 8.1 What We Use

| Technology          | Purpose                 | Duration                              |
| ------------------- | ----------------------- | ------------------------------------- |
| **Session cookies** | Keep you logged in      | Until browser closes or logout        |
| **JWT tokens**      | Authentication          | 15 minutes (access), 7 days (refresh) |
| **Local storage**   | Game preferences, theme | Until cleared                         |

### 8.2 What We Don't Use

- Third-party tracking cookies
- Advertising cookies
- Social media tracking pixels
- Analytics services that track individual users

### 8.3 Managing Cookies

You can disable cookies in your browser settings, but this may prevent the Service from functioning properly.

---

## 9. Children's Privacy

RingRift is not intended for children under 13 years of age. We do not knowingly collect personal information from children under 13.

If you believe we have collected data from a child under 13, please contact us immediately at [your-email@example.com], and we will delete the information.

---

## 10. International Users

If you access the Service from outside [Your Country], please be aware that your data may be transferred to and processed in [Your Country]. By using the Service, you consent to this transfer.

---

## 11. Changes to This Policy

We may update this Privacy Policy from time to time. Changes will be posted on this page with an updated "Last Updated" date.

For significant changes, we will make reasonable efforts to notify you (e.g., via email or in-app notification).

---

## 12. Contact Us

For privacy-related questions or to exercise your rights, contact us at:

- **Email:** [your-email@example.com]
- **GitHub:** [your-github-repo-url]

We aim to respond to all requests within 30 days.

---

## 13. Open Source Transparency

RingRift is open source. You can review exactly how we handle data by examining our source code:

- **Authentication:** `src/server/routes/auth.ts`
- **Data storage:** `prisma/schema.prisma`
- **Data export:** `src/server/routes/user.ts` (data export endpoint)
- **Data deletion:** `src/server/routes/user.ts` (account deletion endpoint)
- **Logging:** `src/server/utils/logger.ts`

We believe in transparency and welcome security researchers to review our code.

---

## Summary

| Question              | Answer                        |
| --------------------- | ----------------------------- |
| Do you sell my data?  | **No**                        |
| What do you collect?  | Email, username, game history |
| Can I delete my data? | **Yes**, anytime in Settings  |
| Can I export my data? | **Yes**, anytime in Settings  |
| Do you use tracking?  | **No** third-party tracking   |
| Is my password safe?  | **Yes**, hashed with bcrypt   |
