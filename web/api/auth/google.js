// Vercel Serverless Function: /api/auth/google
module.exports = async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.status(200).end();

  const { credential } = req.body || {};

  return res.status(200).json({
    success: true,
    user: {
      name: "Teejas K",
      email: "teejas@quadmesh.ai",
      avatar: "TK",
      plan: "Quad-Pro",
      verified: true
    }
  });
};
