// static/js/app.js
const form = document.getElementById('upload-form');
const imageInput = document.getElementById('image');
const resultDiv = document.getElementById('result');
const resultJSON = document.getElementById('result-json');
const msg = document.getElementById('msg');
const submitBtn = document.getElementById('submit-btn');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!imageInput.files || imageInput.files.length === 0) {
    msg.textContent = "Pilih file gambar dulu.";
    return;
  }
  const file = imageInput.files[0];
  const formData = new FormData();
  formData.append('image', file);

  submitBtn.disabled = true;
  msg.textContent = "Mengirim gambar...";

  try {
    const resp = await fetch('/predict', {
      method: 'POST',
      body: formData
    });
    const data = await resp.json();
    if (!resp.ok) {
      msg.textContent = data.error || "Terjadi kesalahan.";
      submitBtn.disabled = false;
      return;
    }
    resultDiv.hidden = false;
    resultJSON.textContent = JSON.stringify(data, null, 2);
    msg.textContent = "Selesai.";
  } catch (err) {
    msg.textContent = "Koneksi gagal.";
  } finally {
    submitBtn.disabled = false;
  }
});
