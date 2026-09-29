// Vercel Serverless Function: /api/payment/create-order
// Supports Quad-Plus (300 credits @ ₹699), Quad-Pro (1500 credits @ ₹1,899), Quad-Max (8000 credits @ ₹7,999), Quad-Enterprise (Pay-As-You-Use)

const PRICING = {
  "quad-plus": { name: "Quad-Plus", amountPaise: 69900, amountINR: 699, credits: 300 },
  "quad-pro": { name: "Quad-Pro", amountPaise: 189900, amountINR: 1899, credits: 1500 },
  "quad-max": { name: "Quad-Max", amountPaise: 799900, amountINR: 7999, credits: 8000 },
  "quad-enterprise": { name: "Quad-Enterprise", amountPaise: 0, amountINR: 0, payAsUse: true, credits: 0 }
};

module.exports = async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.status(200).end();

  const { plan_id = "quad-pro", user_email = "user@quadmesh.ai" } = req.body || {};
  const planKey = (plan_id || "").toLowerCase().trim();
  const plan = PRICING[planKey] || PRICING["quad-pro"];

  const orderId = `order_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
  const keyId = process.env.RAZORPAY_KEY_ID || "rzp_test_QuadmeshDemoKey";

  return res.status(200).json({
    order_id: orderId,
    amount: plan.amountPaise,
    currency: "INR",
    plan_name: plan.name,
    credits: plan.credits,
    key_id: keyId,
    user_email: user_email
  });
};
