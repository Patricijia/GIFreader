import { ChakraProvider } from '@chakra-ui/react';
import FileUpload from './fileupload';

function Main() {
    return (
        <ChakraProvider>
            <FileUpload></FileUpload>
        </ChakraProvider>    
    )
}

export default Main;