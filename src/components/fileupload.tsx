import { SyntheticEvent, useState } from 'react'

import {
    Box,
    Text,
    Flex,    
    Input,
    Image    
} from '@chakra-ui/react'

import { validateFileSize, validateFileType } from '../service/fileValidatorService';
import FileService from '../service/fileService';
import GifDescription from '../model/GifDescription';

function FileUpload() {
    const [uploadFormError, setUploadFormError] = useState<string>('');
    const [imageDescriptionResult, setImageDescriptionResult] = useState<GifDescription>();

    const handleFileUpload = async (element: HTMLInputElement) => {
        const file = element.files;

        if (!file) {
            return;
        }
        
        const validFileSize = await validateFileSize(file[0].size);
        const validFileType = await validateFileType(FileService.getFileExtension(file[0].name));

        if (!validFileSize.isValid) {
            setUploadFormError(validFileSize.errorMessage);
            return;
        }

        if (!validFileType.isValid) {
            setUploadFormError(validFileType.errorMessage);
            return;
        }
        
        if (uploadFormError && validFileSize.isValid) {
            setUploadFormError('');
        }

        const fileService = new FileService(file[0]);
        const fileUploadResponse = await fileService.uploadFile();

        element.value = '';

        if(fileUploadResponse.success)
        {
            setImageDescriptionResult(fileUploadResponse.description);
        }       
    }

    return (
        <div>
        <Box
            width="50%"
            m="100px auto"
            padding="2"
            shadow="base"
        >
            <Flex
                direction="column"
                alignItems="center"
                mb="5"
            >
                <Text fontSize="2xl" mb="4">Upload a File</Text>
                {
                    uploadFormError &&
                    <Text mt="5" color="red">{uploadFormError}</Text>
                }
                <Box
                    mt="10"
                    ml="24"
                >
                    <Input
                        type="file"
                        variant="unstyled"
                        onChange={(e: SyntheticEvent) => handleFileUpload(e.currentTarget as HTMLInputElement)}
                    />
                </Box>
            </Flex>            
        </Box>        
        {
            imageDescriptionResult &&
            <Box width="50%"
                m="100px auto"
                padding="2"            
                shadow="base">
                <Flex direction="column"
                    alignItems="center"
                    mb="5">
                    <Image src={imageDescriptionResult?.imageUri ?? ""} alt={imageDescriptionResult.description + ". Detected text in image says - " + imageDescriptionResult.detectedText}></Image> 
                    <Text mt="5" color="red"><b>Image Description added to gif - </b>{imageDescriptionResult.description}</Text>
                    <Text mt="5" color="green"><b>Image Text detected in gif image -</b> {imageDescriptionResult.detectedText}</Text>
                </Flex>
            </Box>
        }
        </div>
    )
}

export default FileUpload