const pageLines = document.getElementsByClassName("page_line");
const canvas = document.getElementsByClassName("blank-first__canvas")[0];
const ctx = canvas.getContext("2d");


const emptyImage = [
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAwoAAAAyCAYAAAD8+Sx8AAADaklEQVR4Xu3ZsRHDMAwEQav/LtkIndqj4Ar4VYwECyY3es459+MjQIAAAQIECBAgQIDAj8Bz7xUKngQBAgQIECBAgAABAn8CQsGDIECAAAECBAgQIEDgJSAUPAoCBAgQIECAAAECBISCN0CAAAECBAgQIECAQAv4o9BGJggQIECAAAECBAjMCQiFuZNbmAABAgQIECBAgEALCIU2MkGAAAECBAgQIEBgTkAozJ3cwgQIECBAgAABAgRaQCi0kQkCBAgQIECAAAECcwJCYe7kFiZAgAABAgQIECDQAkKhjUwQIECAAAECBAgQmBMQCnMntzABAgQIECBAgACBFhAKbWSCAAECBAgQIECAwJyAUJg7uYUJECBAgAABAgQItIBQaCMTBAgQIECAAAECBOYEhMLcyS1MgAABAgQIECBAoAWEQhuZIECAAAECBAgQIDAnIBTmTm5hAgQIECBAgAABAi0gFNrIBAECBAgQIECAAIE5AaEwd3ILEyBAgAABAgQIEGgBodBGJggQIECAAAECBAjMCQiFuZNbmAABAgQIECBAgEALCIU2MkGAAAECBAgQIEBgTkAozJ3cwgQIECBAgAABAgRaQCi0kQkCBAgQIECAAAECcwJCYe7kFiZAgAABAgQIECDQAkKhjUwQIECAAAECBAgQmBMQCnMntzABAgQIECBAgACBFhAKbWSCAAECBAgQIECAwJyAUJg7uYUJECBAgAABAgQItIBQaCMTBAgQIECAAAECBOYEhMLcyS1MgAABAgQIECBAoAWEQhuZIECAAAECBAgQIDAnIBTmTm5hAgQIECBAgAABAi0gFNrIBAECBAgQIECAAIE5AaEwd3ILEyBAgAABAgQIEGgBodBGJggQIECAAAECBAjMCQiFuZNbmAABAgQIECBAgEALCIU2MkGAAAECBAgQIEBgTkAozJ3cwgQIECBAgAABAgRaQCi0kQkCBAgQIECAAAECcwJCYe7kFiZAgAABAgQIECDQAkKhjUwQIECAAAECBAgQmBMQCnMntzABAgQIECBAgACBFhAKbWSCAAECBAgQIECAwJyAUJg7uYUJECBAgAABAgQItIBQaCMTBAgQIECAAAECBOYEhMLcyS1MgAABAgQIECBAoAWec87tMRMECBAgQIAAAQIECCwJfAHjtsa9AmRVxAAAAABJRU5ErkJggg==",
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAwoAAAAyCAYAAAD8+Sx8AAADZ0lEQVR4Xu3ZsRHDMAwEQav/LtkIndqj4Ar4VYwECyY3es459+MjQIAAAQIECBAgQIDAj8Bz7xUKngQBAgQIECBAgAABAn8CQsGDIECAAAECBAgQIEDgJSAUPAoCBAgQIECAAAECBISCN0CAAAECBAgQIECAQAv4o9BGJggQIECAAAECBAjMCQiFuZNbmAABAgQIECBAgEALCIU2MkGAAAECBAgQIEBgTkAozJ3cwgQIECBAgAABAgRaQCi0kQkCBAgQIECAAAECcwJCYe7kFiZAgAABAgQIECDQAkKhjUwQIECAAAECBAgQmBMQCnMntzABAgQIECBAgACBFhAKbWSCAAECBAgQIECAwJyAUJg7uYUJECBAgAABAgQItIBQaCMTBAgQIECAAAECBOYEhMLcyS1MgAABAgQIECBAoAWEQhuZIECAAAECBAgQIDAnIBTmTm5hAgQIECBAgAABAi0gFNrIBAECBAgQIECAAIE5AaEwd3ILEyBAgAABAgQIEGgBodBGJggQIECAAAECBAjMCQiFuZNbmAABAgQIECBAgEALCIU2MkGAAAECBAgQIEBgTkAozJ3cwgQIECBAgAABAgRaQCi0kQkCBAgQIECAAAECcwJCYe7kFiZAgAABAgQIECDQAkKhjUwQIECAAAECBAgQmBMQCnMntzABAgQIECBAgACBFhAKbWSCAAECBAgQIECAwJyAUJg7uYUJECBAgAABAgQItIBQaCMTBAgQIECAAAECBOYEhMLcyS1MgAABAgQIECBAoAWEQhuZIECAAAECBAgQIDAnIBTmTm5hAgQIECBAgAABAi0gFNrIBAECBAgQIECAAIE5AaEwd3ILEyBAgAABAgQIEGgBodBGJggQIECAAAECBAjMCQiFuZNbmAABAgQIECBAgEALCIU2MkGAAAECBAgQIEBgTkAozJ3cwgQIECBAgAABAgRaQCi0kQkCBAgQIECAAAECcwJCYe7kFiZAgAABAgQIECDQAkKhjUwQIECAAAECBAgQmBMQCnMntzABAgQIECBAgACBFhAKbWSCAAECBAgQIECAwJyAUJg7uYUJECBAgAABAgQItIBQaCMTBAgQIECAAAECBOYEhMLcyS1MgAABAgQIECBAoAWEQhuZIECAAAECBAgQIDAn8AUFQscU4LM33gAAAABJRU5ErkJggg==',
];
const lineOffset = [];
let lineHeight = 0;



function initCanvas(lines = 10) {
  //setup canvas
  ctx.lineWidth = 1;
  canvas.style.width = "90%";
  canvas.style.height = "90%";
  canvas.width = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
  ctx.strokeStyle = "#c4c4c4";

  const distance = canvas.height - 0.5;
  lineHeight = Math.round(distance / lines);
  let cOffset = 0;

  for (let i = 0; i < lines; i++) {
    lineOffset.push(cOffset + 0.5);
    ctx.moveTo(0, cOffset + 0.5);
    ctx.lineTo(canvas.width + 0.5, cOffset + 0.5);
    ctx.stroke();
    cOffset += lineHeight;
  }

  //last line
  ctx.moveTo(0, canvas.height - 0.5);
  ctx.lineTo(canvas.width + 0.5, canvas.height);
  ctx.stroke();

  lineOffset.push(canvas.height - 0.5);
}

initCanvas();

ctx.strokeStyle = "#000";
let drawingMode = false;
let prevX = 0;
let prevY = 0;

function initDraw(e) {
  prevX = e.offsetX;
  prevY = e.offsetY;
  drawingMode = true;
}

function drawOnCanvas(event) {
  if (!drawingMode) return;
  const xCor = event.offsetX;
  const yCor = event.offsetY;
  ctx.beginPath();
  ctx.moveTo(prevX, prevY);
  ctx.quadraticCurveTo(prevX, prevY, xCor, yCor);
  ctx.stroke();
  prevX = xCor;
  prevY = yCor;
}

function getBase64Images() {}

// canvas.addEventListener("mousedown", (e) => initDraw(e));
// canvas.addEventListener("mousemove", (e) => drawOnCanvas(e));
// canvas.addEventListener("touchmove", (e) => drawOnCanvas(e));
// canvas.addEventListener("mouseup", () => (drawingMode = false));
// canvas.addEventListener("mouseout", () => (drawingMode = false));
// canvas.addEventListener("touchend", () => (drawingMode = false));
canvas.addEventListener("pointerup", () => (drawingMode = false));
canvas.addEventListener("pointerdown", (e) => initDraw(e));
canvas.addEventListener("pointermove", (e) => drawOnCanvas(e));

function cropBase64(canvas, offsetX, offsetY, width, height, callback) {
  // create an in-memory canvas
  var buffer = document.createElement("canvas");
  var b_ctx = buffer.getContext("2d");
  // set its width/height to the required ones
  buffer.width = width;
  buffer.height = height;
  b_ctx.fillStyle = "white";
  b_ctx.fillRect(0, 0, width, height);
  // draw the main canvas on our buffer one
  // drawImage(source, source_X, source_Y, source_Width, source_Height,
  //  dest_X, dest_Y, dest_Width, dest_Height)
  b_ctx.drawImage(
    canvas,
    offsetX,
    offsetY,
    width,
    height,
    0,
    0,
    buffer.width,
    buffer.height
  );
  // now call the callback with the dataURL of our buffer canvas
  callback(buffer.toDataURL());
}

function submit() {
  const data = {};
  for (let i = 0; i < lineOffset.length; i++) {
    cropBase64(canvas, 0, lineOffset[i], canvas.width, lineHeight, (base64) => {
      if (!emptyImage.includes(base64)) {
        data[i] = {
          index: i,
          image: base64,
        };
      }
    });
  }
  postData(data);
}

function postData(data) {
  fetch(`${window.origin}/upload`, {
    method: "POST",
    credentials: "include",
    body: JSON.stringify(data),
    cache: "no-cache",
    headers: new Headers({
      "content-type": "application/json",
    }),
  })
    .then(function (response) {
      response.json().then((data) => {
        processDataToHTML(data);
      });
    })
    .catch(function (error) {
      console.log("Fetch error: " + error);
    });
}

function processDataToHTML(data) {
  const {predict,index} = data;
  pageLines[index].innerHTML = predict;
}

function disableScroll() {
  document.body.addEventListener("touchmove", (e) => e.preventDefault(), {
    passive: false,
  });
}
function enableScroll() {
  document.body.removeEventListener("touchmove", (e) => e.preventDefault());
}

disableScroll();
window.submit = submit;
