from pdf2image import convert_from_path

pdf_path = 'Nhập đường dẫn file pdf cần chuyển đổi sang ảnh tại đây'
img_fol = 'Nhập đường dẫn thư mục chứa ảnh đầu ra ở đây'

images = convert_from_path(pdf_path, poppler_path='')
name = pdf_path.split('/')[-1].replace('.pdf', '')

for i in range(len(images)):
    images[i].save(img_fol + '/' + name + '_page' + str(i + 1) + '.jpg')
