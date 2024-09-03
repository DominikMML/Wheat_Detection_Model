//MIT License
//Copyright 2023 Dominik Mielczarek
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:
//The above copyright notice and this permission notice shall be included in all
//copies or substantial portions of the Software.
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.
//
function ReadImage(){
    if ($("#image-previewPredykcja")[0]) {
        $("#image-previewPredykcja").hide();
    } else {
        $("#image-previewPredykcja").show();
    }
    // Show preview image for select file before upload
    $('#imagePreview').attr('src', URL.createObjectURL(event.target.files[0]));
}
// Upload image using ajax
$('#upload').click(function(){
    // Create form data
    var formData = new FormData();
    // add file to form data
    formData.append('file', $('#fileInput')[0].files[0]);
    $.ajax({
        url: '/api/upload', // API Endpoint
        type: 'POST', // Request type
        data: formData, // Request data
        contentType: false,
        processData: false,
        success: function(data){
            // On request succss, we show image from server
            $('#imagePreview').attr('src', data);
        }
    });
});