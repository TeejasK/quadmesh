// Vercel Serverless Function: /api/bank/settings
let bankState = {
  holder_name: "Teejas K",
  bank_name: "HDFC Bank",
  account_number_masked: "••••••••0394",
  ifsc_code: "HDFC0000240",
  upi_id: "quadmesh@hdfcbank",
  settlement_cycle: "T+1 Daily Auto-Credit",
  status: "verified_active"
};

module.exports = async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.status(200).end();

  if (req.method === 'POST') {
    const { holder_name, bank_name, account_number, ifsc_code, upi_id } = req.body || {};
    bankState = {
      ...bankState,
      holder_name: holder_name || bankState.holder_name,
      bank_name: bank_name || bankState.bank_name,
      account_number_masked: account_number ? ("••••••••" + account_number.slice(-4)) : bankState.account_number_masked,
      ifsc_code: ifsc_code || bankState.ifsc_code,
      upi_id: upi_id || bankState.upi_id
    };
    return res.status(200).json({ success: true, settings: bankState });
  }

  return res.status(200).json(bankState);
};
